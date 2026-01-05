"""
Fine-tune model from checkpoint with dense (per-frame) predictions

This script:
1. Loads the current trained model checkpoint
2. Freezes transformer encoder (or uses very low LR)
3. Retrains classifier head for dense predictions
4. Trains for 30 epochs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from tqdm import tqdm
import sys
import os
from pathlib import Path
import logging
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import cv2
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

current_dir = Path(__file__).parent.parent.absolute()
models_dir = current_dir / 'models'
sys.path.insert(0, str(models_dir))

try:
    from actionnet_transformer_video import ActionNetTransformerVideo
    from train_transformer_video import SimpleVideoDataset, train_epoch, validate, ACTIONS
except ImportError as e:
    logger.error(f"Failed to import: {e}")
    raise

def fine_tune_model(
    checkpoint_path: str,
    train_file: str,
    val_file: str,
    extracted_frames_dir: str = None,
    output_dir: str = "models",
    num_epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 0.0001,
    encoder_lr: float = 0.00005,
    classifier_lr: float = 0.0005,
    freeze_encoder: bool = False,
    img_size: int = 224,
    num_frames: int = 30,
    device: str = None
):
    """
    Fine-tune model from checkpoint with dense predictions

    Args:
        checkpoint_path: Path to checkpoint file
        train_file: Path to train.json
        val_file: Path to val.json
        extracted_frames_dir: Directory with pre-extracted frames
        output_dir: Output directory for models
        num_epochs: Number of fine-tuning epochs
        batch_size: Batch size
        learning_rate: Learning rate (if freeze_encoder=False, this is used for all)
        encoder_lr: Learning rate for encoder (fine-tuning)
        classifier_lr: Learning rate for classifier (normal training)
        freeze_encoder: If True, completely freeze encoder weights
        img_size: Image size
        num_frames: Number of frames per sequence
        device: Device to use
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(output_dir, exist_ok=True)


    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)


    logger.info("Initializing model...")
    model = ActionNetTransformerVideo(
        img_size=img_size,
        patch_size=16,
        num_frames=num_frames,
        hidden_size=128,
        num_layers=3,
        num_heads=8,
        num_actions=9,
        dropout=0.1
    ).to(device)


    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Checkpoint validation accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")


    model.prediction_mode = 'dense'
    logger.info("Model set to dense prediction mode")


    if freeze_encoder:
        logger.info("Freezing transformer encoder...")
        for name, param in model.named_parameters():
            if 'transformer_encoder' in name or 'frame_embedder' in name or 'frame_projection' in name or 'pos_encoder' in name:
                param.requires_grad = False
                logger.info(f"  Frozen: {name}")
    else:
        logger.info("Using differential learning rates (encoder: low, classifier: normal)")


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")


    if extracted_frames_dir:
        train_frames_dir = os.path.join(extracted_frames_dir, 'train')
        val_frames_dir = os.path.join(extracted_frames_dir, 'val')
        logger.info(f"Using pre-extracted frames from: {extracted_frames_dir}")
        train_dataset = SimpleVideoDataset(train_file, extracted_frames_dir=train_frames_dir, img_size=img_size, num_frames=num_frames)
        val_dataset = SimpleVideoDataset(val_file, extracted_frames_dir=val_frames_dir, img_size=img_size, num_frames=num_frames)
    else:
        raise ValueError("extracted_frames_dir must be provided")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")


    if freeze_encoder:

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=classifier_lr,
            weight_decay=1e-5
        )
    else:

        encoder_params = []
        classifier_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'transformer_encoder' in name or 'frame_embedder' in name or 'frame_projection' in name or 'pos_encoder' in name:
                    encoder_params.append(param)
                else:
                    classifier_params.append(param)

        optimizer = optim.Adam([
            {'params': encoder_params, 'lr': encoder_lr, 'weight_decay': 1e-5},
            {'params': classifier_params, 'lr': classifier_lr, 'weight_decay': 1e-5}
        ])
        logger.info(f"Encoder LR: {encoder_lr}, Classifier LR: {classifier_lr}")



    class_weights = torch.ones(len(ACTIONS)).to(device)
    class_weights[8] = 1.0 / 29.0
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    logger.info(f"Using class weights: {class_weights.cpu().numpy()}")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )


    best_val_acc = checkpoint.get('val_acc', 0.0)
    best_model_path = os.path.join(output_dir, 'best_model_dense_finetuned.pth')
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    logger.info("="*60)
    logger.info("Starting fine-tuning with dense predictions...")
    logger.info("="*60)

    for epoch in range(num_epochs):
        logger.info(f"\nFine-tuning Epoch {epoch+1}/{num_epochs}")
        logger.info("-" * 50)


        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, use_dense=True)


        val_loss, val_acc, predictions, labels = validate(model, val_loader, criterion, device, use_dense=True)


        from collections import Counter
        pred_counts = Counter(predictions)
        label_counts = Counter(labels)
        logger.info(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        logger.info(f"  Prediction distribution: {dict(pred_counts)}")
        logger.info(f"  Label distribution: {dict(label_counts)}")

        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)


        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'training_history': training_history,
                'fine_tuned_from': checkpoint_path
            }, best_model_path)
            logger.info(f"✓ Best validation accuracy improved to {best_val_acc:.2f}%. Model saved.")

        scheduler.step(val_loss)

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    logger.info("\n" + "="*50)
    logger.info("Fine-tuning Complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Model saved to: {best_model_path}")
    logger.info("="*50)


    logger.info("Loading best model for final evaluation...")
    best_checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.eval()

    logger.info("Final evaluation on validation set...")
    val_loss, val_acc, predictions, labels = validate(model, val_loader, criterion, device, use_dense=True)


    logger.info("\n" + "="*50)
    logger.info("Classification Report:")
    logger.info("="*50)
    report = classification_report(
        labels, predictions,
        target_names=ACTIONS,
        digits=4
    )
    logger.info(report)

    return model, training_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune model with dense predictions')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file to fine-tune from')
    parser.add_argument('--train-file', type=str, required=True,
                        help='Path to train.json')
    parser.add_argument('--val-file', type=str, required=True,
                        help='Path to val.json')
    parser.add_argument('--extracted-frames-dir', type=str, required=True,
                        help='Directory containing pre-extracted frames')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--encoder-lr', type=float, default=0.00001,
                        help='Learning rate for encoder (fine-tuning)')
    parser.add_argument('--classifier-lr', type=float, default=0.0001,
                        help='Learning rate for classifier')
    parser.add_argument('--freeze-encoder', action='store_true',
                        help='Completely freeze encoder weights')
    args = parser.parse_args()

    fine_tune_model(
        checkpoint_path=args.checkpoint,
        train_file=args.train_file,
        val_file=args.val_file,
        extracted_frames_dir=args.extracted_frames_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        encoder_lr=args.encoder_lr,
        classifier_lr=args.classifier_lr,
        freeze_encoder=args.freeze_encoder
    )
