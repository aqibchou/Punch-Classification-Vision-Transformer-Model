"""
Train ActionNet Transformer on Raw Video Frames

This script trains the transformer directly on raw video frames
without pose estimation.
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
from typing import List, Tuple
from scipy.signal import find_peaks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

current_dir = Path(__file__).parent.parent.absolute()
models_dir = current_dir / 'models'
sys.path.insert(0, str(models_dir))

try:
    from actionnet_transformer_video import ActionNetTransformerVideo
except ImportError as e:
    logger.error(f"Failed to import ActionNetTransformerVideo: {e}")
    raise

ACTIONS = [
    "head_hit_left", "head_hit_right",
    "body_hit_left", "body_hit_right",
    "block_left", "block_right",
    "miss_left", "miss_right",
    "no_action"
]


class SimpleVideoDataset(Dataset):
    """Video dataset that loads from pre-extracted frames (fast) or video files (slow)"""

    def __init__(self, json_file, video_base_dir=None, extracted_frames_dir=None, img_size=224, num_frames=30):
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.video_base_dir = video_base_dir
        self.extracted_frames_dir = extracted_frames_dir
        self.img_size = img_size
        self.num_frames = num_frames
        self.use_extracted = extracted_frames_dir is not None

        if self.use_extracted:
            self.frame_files = {}
            if os.path.exists(extracted_frames_dir):
                for filename in os.listdir(extracted_frames_dir):
                    if filename.endswith('.npy'):
                        parts = filename.replace('.npy', '').split('_')
                        if len(parts) >= 4:
                            idx = int(parts[0])
                            video_id = '_'.join(parts[1:-2])
                            frame_num = int(parts[-2])
                            key = (video_id, frame_num)
                            self.frame_files[key] = os.path.join(extracted_frames_dir, filename)
            logger.info(f"Found {len(self.frame_files)} pre-extracted frame files")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        video_id = item['video_id']
        frame_number = item['frame_number']
        label = item['label']

        if self.use_extracted:
            key = (video_id, frame_number)
            if key in self.frame_files:
                frames = np.load(self.frame_files[key])
            else:
                frame_file = os.path.join(self.extracted_frames_dir, f"{idx:06d}_{video_id}_{frame_number:06d}_{label}.npy")
                if os.path.exists(frame_file):
                    frames = np.load(frame_file)
                else:
                    raise FileNotFoundError(f"Pre-extracted frame not found for {video_id} frame {frame_number}")
        else:
            video_path = self._find_video(video_id)
            frames = self._extract_frames(video_path, frame_number)

        frames_tensor = torch.stack([self.transform(frame) for frame in frames])

        return frames_tensor, label

    def _find_video(self, video_id):
        """Find video file path"""
        import glob

        patterns = [
            f"{self.video_base_dir}/*/{video_id}/data/*.mp4",
            f"{self.video_base_dir}/*/{video_id}/data/*.MOV",
            f"{self.video_base_dir}/{video_id}/data/*.mp4",
            f"{self.video_base_dir}/{video_id}/data/*.MOV",
        ]

        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                return matches[0]

        raise FileNotFoundError(f"Video not found for {video_id}")

    def _extract_frames(self, video_path, center_frame):
        """Extract sequence of frames around center_frame"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = max(0, center_frame - self.num_frames // 2)
        end_frame = min(total_frames, center_frame + self.num_frames // 2)

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

        while len(frames) < self.num_frames:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))

        frames = frames[:self.num_frames]

        return frames


def train_epoch(model, train_loader, criterion, optimizer, device, use_dense=True):
    """Train for one epoch with dense (per-frame) or sequence-level predictions"""
    model.train()
    model.prediction_mode = 'dense' if use_dense else 'cls'

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for frames, labels in tqdm(train_loader, desc="Training"):
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        result = model(frames, prediction_mode='dense' if use_dense else 'cls')
        outputs = result['output']

        if use_dense:
            batch_size, seq_len = outputs.shape[0], outputs.shape[1]
            center_frame = seq_len // 2

            per_frame_labels = torch.full((batch_size, seq_len), 8, dtype=torch.long, device=device)
            per_frame_labels[:, center_frame] = labels

            outputs_flat = outputs.view(-1, outputs.shape[-1])
            labels_flat = per_frame_labels.view(-1)

            loss = criterion(outputs_flat, labels_flat)

            center_predictions = outputs[:, center_frame, :]
            _, predicted = torch.max(center_predictions.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        else:
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        train_loss += loss.item()

    avg_loss = train_loss / len(train_loader)
    accuracy = 100 * train_correct / train_total if train_total > 0 else 0.0

    return avg_loss, accuracy


def validate(model, val_loader, criterion, device, use_dense=True):
    """Validate model with dense (per-frame) or sequence-level predictions"""
    model.eval()
    model.prediction_mode = 'dense' if use_dense else 'cls'

    val_loss = 0.0
    val_correct = 0
    val_total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for frames, labels in tqdm(val_loader, desc="Validating"):
            frames = frames.to(device)
            labels = labels.to(device)

            result = model(frames, prediction_mode='dense' if use_dense else 'cls')
            outputs = result['output']

            if use_dense:
                batch_size, seq_len = outputs.shape[0], outputs.shape[1]
                center_frame = seq_len // 2

                per_frame_labels = torch.full((batch_size, seq_len), 8, dtype=torch.long, device=device)
                per_frame_labels[:, center_frame] = labels

                outputs_flat = outputs.view(-1, outputs.shape[-1])
                labels_flat = per_frame_labels.view(-1)
                loss = criterion(outputs_flat, labels_flat)

                center_predictions = outputs[:, center_frame, :]
                _, predicted = torch.max(center_predictions.data, 1)
            else:
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)

            val_loss += loss.item()
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * val_correct / val_total if val_total > 0 else 0.0

    return avg_loss, accuracy, all_predictions, all_labels


def detect_action_peaks(frame_predictions: np.ndarray,
                       frame_confidences: np.ndarray,
                       min_peak_gap: int = 5,
                       min_peak_height: float = 0.5,
                       min_action_duration: int = 2,
                       merge_threshold: int = 5) -> List[Tuple[int, int, int]]:
    """
    Detect actions using peak detection in confidence scores.
    This separates multiple actions of the same class (e.g., double jabs).

    Args:
        frame_predictions: (num_frames,) array of predicted class IDs
        frame_confidences: (num_frames, num_classes) array of confidence scores (softmax probabilities)
        min_peak_gap: Minimum frames between peaks to count as separate actions
        min_peak_height: Minimum confidence to count as a peak
        min_action_duration: Minimum frames for a valid action
        merge_threshold: Maximum gap between actions to merge (for different classes)

    Returns:
        List of (start_frame, end_frame, class_id) tuples
    """
    if len(frame_predictions) == 0 or frame_confidences.shape[0] == 0:
        return []

    num_frames, num_classes = frame_confidences.shape
    all_detections = []


    for class_id in range(8):

        class_confidences = frame_confidences[:, class_id]


        peaks, properties = find_peaks(
            class_confidences,
            distance=min_peak_gap,
            height=min_peak_height,
            prominence=0.2
        )


        for peak_idx in peaks:
            peak_conf = class_confidences[peak_idx]



            threshold = max(peak_conf * 0.1, 0.05)


            start = peak_idx
            while start > 0 and class_confidences[start - 1] > threshold:
                start -= 1


            end = peak_idx
            while end < num_frames - 1 and class_confidences[end + 1] > threshold:
                end += 1



            if end - start < 2:
                start = max(0, peak_idx - 1)
                end = min(num_frames - 1, peak_idx + 1)


            duration = end - start + 1
            if duration >= min_action_duration:
                all_detections.append((start, end, class_id))


    all_detections.sort(key=lambda x: x[0])


    if len(all_detections) < 2:
        return all_detections

    merged_detections = []
    i = 0
    while i < len(all_detections):
        current = all_detections[i]
        start, end, class_id = current


        j = i + 1
        while j < len(all_detections):
            next_start, next_end, next_class = all_detections[j]


            if next_start <= end:

                current_peak_idx = start + (end - start) // 2
                next_peak_idx = next_start + (next_end - next_start) // 2
                current_peak_conf = frame_confidences[current_peak_idx, class_id]
                next_peak_conf = frame_confidences[next_peak_idx, next_class]

                if next_peak_conf > current_peak_conf:

                    break
                else:

                    j += 1
                    continue
            else:

                gap = next_start - end - 1


                if next_class == class_id and gap <= merge_threshold:

                    gap_start = end + 1
                    gap_end = next_start - 1
                    if gap_start <= gap_end:

                        gap_confidences = frame_confidences[gap_start:gap_end+1, class_id]
                        min_gap_conf = np.min(gap_confidences) if len(gap_confidences) > 0 else 1.0


                        current_peak_idx = start + (end - start) // 2
                        next_peak_idx = next_start + (next_end - next_start) // 2
                        current_peak_conf = frame_confidences[current_peak_idx, class_id]
                        next_peak_conf = frame_confidences[next_peak_idx, next_class]
                        avg_peak_conf = (current_peak_conf + next_peak_conf) / 2


                        if min_gap_conf < avg_peak_conf * 0.4:

                            break
                        else:

                            end = next_end
                            j += 1
                            continue
                    else:

                        end = next_end
                        j += 1
                        continue
                else:

                    break

        merged_detections.append((start, end, class_id))
        i = j

    return merged_detections


def detect_actions_from_model_output(model_output: torch.Tensor,
                                     use_peak_detection: bool = True,
                                     min_peak_gap: int = 5,
                                     min_peak_height: float = 0.5,
                                     min_action_duration: int = 2,
                                     merge_threshold: int = 5) -> List[Tuple[int, int, int]]:
    """
    Detect actions from model output using peak detection or temporal smoothing.

    Args:
        model_output: (batch, seq_len, num_actions) or (seq_len, num_actions) tensor of logits
        use_peak_detection: If True, use peak detection (recommended). If False, use temporal smoothing.
        min_peak_gap: Minimum frames between peaks (for peak detection)
        min_peak_height: Minimum confidence for peak (for peak detection)
        min_action_duration: Minimum frames for valid action
        merge_threshold: Maximum gap to merge actions (for temporal smoothing)

    Returns:
        List of (start_frame, end_frame, class_id) tuples for each detected action
    """

    if isinstance(model_output, torch.Tensor):
        model_output = model_output.detach().cpu().numpy()


    if len(model_output.shape) == 3:

        model_output = model_output[0]




    exp_scores = np.exp(model_output - np.max(model_output, axis=-1, keepdims=True))
    frame_confidences = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


    frame_predictions = np.argmax(frame_confidences, axis=-1)

    if use_peak_detection:

        return detect_action_peaks(
            frame_predictions,
            frame_confidences,
            min_peak_gap=min_peak_gap,
            min_peak_height=min_peak_height,
            min_action_duration=min_action_duration,
            merge_threshold=merge_threshold
        )
    else:

        return temporal_smoothing(
            frame_predictions,
            min_duration=min_action_duration,
            merge_threshold=merge_threshold
        )


def temporal_smoothing(predictions: np.ndarray, min_duration: int = 2, merge_threshold: int = 5) -> List[Tuple[int, int, int]]:
    """
    Apply temporal smoothing to per-frame predictions to merge consecutive detections.

    Args:
        predictions: (num_frames,) array of class predictions
        min_duration: Minimum frames for a valid action (filter noise)
        merge_threshold: Maximum gap between actions to merge (frames)

    Returns:
        List of (start_frame, end_frame, class_id) tuples
    """
    if len(predictions) == 0:
        return []

    detections = []
    i = 0

    while i < len(predictions):

        if predictions[i] == 8:
            i += 1
            continue


        current_class = predictions[i]
        start_frame = i


        j = i + 1
        last_action_frame = i

        while j < len(predictions):
            if predictions[j] == current_class:

                last_action_frame = j
                j += 1
            elif predictions[j] == 8:

                gap_start = j

                while j < len(predictions) and predictions[j] == 8:
                    j += 1
                gap_length = j - gap_start


                if j < len(predictions) and predictions[j] == current_class and gap_length <= merge_threshold:

                    last_action_frame = j
                    j += 1
                else:

                    break
            else:

                break


        duration = last_action_frame - start_frame + 1
        if duration >= min_duration:
            detections.append((start_frame, last_action_frame, int(current_class)))


        i = j

    return detections


def train_model(
    train_file: str,
    val_file: str,
    video_base_dir: str = None,
    extracted_frames_dir: str = None,
    output_dir: str = "models",
    num_epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 0.0001,
    img_size: int = 224,
    num_frames: int = 30,
    hidden_size: int = 128,
    num_layers: int = 3,
    num_heads: int = 8,
    dropout: float = 0.1,
    device: str = None
):
    """Train the transformer model on raw video"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    logger.info(f"Using device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(output_dir, exist_ok=True)

    if extracted_frames_dir:
        train_frames_dir = os.path.join(extracted_frames_dir, 'train')
        val_frames_dir = os.path.join(extracted_frames_dir, 'val')
        logger.info(f"Using pre-extracted frames from: {extracted_frames_dir}")
        train_dataset = SimpleVideoDataset(train_file, extracted_frames_dir=train_frames_dir, img_size=img_size, num_frames=num_frames)
        val_dataset = SimpleVideoDataset(val_file, extracted_frames_dir=val_frames_dir, img_size=img_size, num_frames=num_frames)
    else:
        if video_base_dir is None:
            raise ValueError("Either video_base_dir or extracted_frames_dir must be provided")
        logger.info(f"Loading frames from video files: {video_base_dir}")
        train_dataset = SimpleVideoDataset(train_file, video_base_dir=video_base_dir, img_size=img_size, num_frames=num_frames)
        val_dataset = SimpleVideoDataset(val_file, video_base_dir=video_base_dir, img_size=img_size, num_frames=num_frames)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    model = ActionNetTransformerVideo(
        img_size=img_size,
        patch_size=16,
        num_frames=num_frames,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        num_actions=len(ACTIONS),
        dropout=dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, 'best_model_video.pth')
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    logger.info("Starting training...")
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
        logger.info("-" * 50)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, use_dense=True)

        val_loss, val_acc, predictions, labels = validate(model, val_loader, criterion, device, use_dense=True)

        scheduler.step(val_loss)

        epoch_progress = ((epoch + 1) / num_epochs) * 100
        logger.info(f"[Epoch {epoch+1}/{num_epochs} - {epoch_progress:.1f}% Complete]")
        logger.info(f"  Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        logger.info(f"  Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.2f}%")
        logger.info(f"  Best Val Acc: {best_val_acc:.2f}%")

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
                'training_history': training_history
            }, best_model_path)
            logger.info(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    logger.info("\n" + "="*50)
    logger.info("Loading best model for final evaluation...")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

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

    logger.info("\n" + "="*50)
    logger.info("Training Complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Model saved to: {best_model_path}")
    logger.info("="*50)

    return model, training_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ActionNet Transformer on Raw Video')
    parser.add_argument('--train-file', type=str, required=True,
                        help='Path to train.json')
    parser.add_argument('--val-file', type=str, required=True,
                        help='Path to val.json')
    parser.add_argument('--video-dir', type=str, default=None,
                        help='Base directory containing video files (if not using pre-extracted frames)')
    parser.add_argument('--extracted-frames-dir', type=str, default=None,
                        help='Directory containing pre-extracted frames (faster, recommended)')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for models')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size (smaller for video)')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--num-frames', type=int, default=30,
                        help='Number of frames per sequence')
    parser.add_argument('--hidden-size', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--dense-prediction', action='store_true', default=True,
                        help='Use dense (per-frame) predictions instead of sequence-level')
    parser.add_argument('--no-dense-prediction', dest='dense_prediction', action='store_false',
                        help='Disable dense prediction, use sequence-level classification')
    args = parser.parse_args()

    train_model(
        train_file=args.train_file,
        val_file=args.val_file,
        video_base_dir=args.video_dir,
        extracted_frames_dir=args.extracted_frames_dir,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        img_size=args.img_size,
        num_frames=args.num_frames,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads
    )
