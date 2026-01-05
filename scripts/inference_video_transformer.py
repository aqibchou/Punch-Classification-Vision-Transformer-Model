"""
Inference Script for ActionNet Transformer Video Model

This script:
1. Loads the trained model checkpoint
2. Processes video files frame-by-frame
3. Extracts 30-frame sequences with sliding window
4. Runs dense prediction inference
5. Applies peak detection to count actions
6. Outputs punch statistics and timestamps
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
import sys
import logging
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

current_dir = Path(__file__).parent.parent.absolute()
models_dir = current_dir / 'models'
sys.path.insert(0, str(models_dir))

try:
    from actionnet_transformer_video import ActionNetTransformerVideo
    from train_transformer_video import detect_actions_from_model_output, ACTIONS
except ImportError as e:
    logger.error(f"Failed to import necessary modules: {e}")
    logger.error(f"Make sure you're running from the project root directory")
    raise

ACTION_NAMES = ACTIONS


class VideoInference:
    """Inference pipeline for video transformer model"""

    def __init__(self,
                 checkpoint_path: str,
                 img_size: int = 224,
                 num_frames: int = 30,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 device: str = None,
                 min_peak_gap: int = 5,
                 min_peak_height: float = 0.5,
                 min_action_duration: int = 2,
                 class_confidence_boost: dict = None):
        """
        Initialize inference pipeline

        Args:
            checkpoint_path: Path to trained model checkpoint (.pth file)
            img_size: Image size for model input
            num_frames: Number of frames per sequence
            hidden_size: Hidden dimension size
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            device: Device to run inference on ('cuda' or 'cpu')
            min_peak_gap: Minimum frames between peaks for peak detection
            min_peak_height: Minimum confidence for peak detection
            min_action_duration: Minimum frames for valid action
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(device)

        self.device = device
        self.img_size = img_size
        self.num_frames = num_frames
        self.min_peak_gap = min_peak_gap
        self.min_peak_height = min_peak_height
        self.min_action_duration = min_action_duration




        if class_confidence_boost is None:

            self.class_confidence_boost = {
                0: 1.0,
                1: 1.0,
                2: 1.5,
                3: 2.0,
                4: 1.3,
                5: 1.0,
                6: 1.0,
                7: 1.4,
            }
        else:
            self.class_confidence_boost = class_confidence_boost

        logger.info(f"Using device: {device}")
        if device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


        logger.info(f"Loading model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)


        self.model = ActionNetTransformerVideo(
            img_size=img_size,
            patch_size=16,
            num_frames=num_frames,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_actions=len(ACTION_NAMES),
            dropout=dropout
        ).to(device)


        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.prediction_mode = 'dense'

        logger.info(f"Model loaded successfully (from epoch {checkpoint.get('epoch', 'N/A')})")
        logger.info(f"Checkpoint validation accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")


        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_frame_sequence(self, video_path: str, start_frame: int) -> np.ndarray:
        """
        Extract a sequence of frames from video

        Args:
            video_path: Path to video file
            start_frame: Starting frame number

        Returns:
            Array of frames: (num_frames, height, width, channels)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:

                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
            else:

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        cap.release()


        while len(frames) < self.num_frames:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))

        frames = frames[:self.num_frames]
        return np.array(frames, dtype=np.uint8)

    def process_video(self,
                     video_path: str,
                     stride: int = 15,
                     output_file: str = None) -> Dict:
        """
        Process entire video and detect all actions

        Args:
            video_path: Path to video file
            stride: Frames to skip between sequences (15 = 50% overlap)
            output_file: Optional JSON file to save results

        Returns:
            Dictionary with action counts, timestamps, and statistics
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        cap.release()

        logger.info(f"Processing video: {video_path}")
        logger.info(f"Total frames: {total_frames}, FPS: {fps:.2f}, Duration: {duration:.2f}s")


        all_detections = []
        sequence_predictions = []


        num_sequences = (total_frames - self.num_frames) // stride + 1
        logger.info(f"Processing {num_sequences} sequences with stride {stride}")

        with torch.no_grad():
            for seq_idx in tqdm(range(num_sequences), desc="Processing sequences"):
                start_frame = seq_idx * stride
                center_frame = start_frame + self.num_frames // 2


                frames = self.extract_frame_sequence(video_path, start_frame)


                frames_tensor = torch.stack([
                    self.transform(frame) for frame in frames
                ]).unsqueeze(0).to(self.device)


                result = self.model(frames_tensor, prediction_mode='dense')
                outputs = result['output']



                boosted_outputs = outputs[0].clone()
                for class_id, boost_factor in self.class_confidence_boost.items():
                    if class_id < boosted_outputs.shape[1] and boost_factor != 1.0:

                        boosted_outputs[:, class_id] = boosted_outputs[:, class_id] * boost_factor


                detections = detect_actions_from_model_output(
                    boosted_outputs,
                    use_peak_detection=True,
                    min_peak_gap=self.min_peak_gap,
                    min_peak_height=self.min_peak_height,
                    min_action_duration=self.min_action_duration
                )


                for start_rel, end_rel, class_id in detections:
                    start_abs = start_frame + start_rel
                    end_abs = start_frame + end_rel
                    center_abs = start_frame + (start_rel + end_rel) // 2


                    confidence_scores = torch.softmax(outputs[0], dim=-1).cpu().numpy()
                    peak_conf = confidence_scores[start_rel + (end_rel - start_rel) // 2, class_id]

                    all_detections.append({
                        'start_frame': int(start_abs),
                        'end_frame': int(end_abs),
                        'center_frame': int(center_abs),
                        'class_id': int(class_id),
                        'action_name': ACTION_NAMES[class_id],
                        'confidence': float(peak_conf),
                        'start_time': float(start_abs / fps) if fps > 0 else 0.0,
                        'end_time': float(end_abs / fps) if fps > 0 else 0.0,
                        'center_time': float(center_abs / fps) if fps > 0 else 0.0
                    })



        logger.info(f"Found {len(all_detections)} raw detections, removing duplicates...")
        unique_detections = self._remove_duplicates(all_detections)
        logger.info(f"After deduplication: {len(unique_detections)} unique actions")


        action_counts = {name: 0 for name in ACTION_NAMES}
        for det in unique_detections:
            action_counts[det['action_name']] += 1


        total_actions = sum(action_counts.values()) - action_counts['no_action']
        landed_hits = (action_counts['head_hit_left'] + action_counts['head_hit_right'] +
                      action_counts['body_hit_left'] + action_counts['body_hit_right'])
        blocks = action_counts['block_left'] + action_counts['block_right']
        misses = action_counts['miss_left'] + action_counts['miss_right']

        results = {
            'video_path': video_path,
            'total_frames': int(total_frames),
            'fps': float(fps),
            'duration_seconds': float(duration),
            'total_actions': total_actions,
            'action_counts': action_counts,
            'statistics': {
                'landed_hits': landed_hits,
                'blocks': blocks,
                'misses': misses,
                'hit_rate': float(landed_hits / total_actions) if total_actions > 0 else 0.0,
                'block_rate': float(blocks / total_actions) if total_actions > 0 else 0.0,
                'miss_rate': float(misses / total_actions) if total_actions > 0 else 0.0
            },
            'detections': unique_detections
        }


        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {output_file}")

        return results

    def _remove_duplicates(self, detections: List[Dict], overlap_threshold: int = 10) -> List[Dict]:
        """
        Remove duplicate detections from overlapping sequences

        Args:
            detections: List of detection dictionaries
            overlap_threshold: Maximum frame difference to consider as duplicate

        Returns:
            List of unique detections (keeping highest confidence)
        """
        if not detections:
            return []


        detections_sorted = sorted(detections, key=lambda x: x['center_frame'])

        unique = []
        for det in detections_sorted:

            is_duplicate = False
            for existing in unique:
                if (det['class_id'] == existing['class_id'] and
                    abs(det['center_frame'] - existing['center_frame']) < overlap_threshold):

                    if det['confidence'] > existing['confidence']:
                        unique.remove(existing)
                        unique.append(det)
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(det)


        unique.sort(key=lambda x: x['center_frame'])
        return unique

    def print_summary(self, results: Dict):
        """Print a summary of results"""
        print("\n" + "="*60)
        print("VIDEO ANALYSIS SUMMARY")
        print("="*60)
        print(f"Video: {Path(results['video_path']).name}")
        print(f"Duration: {results['duration_seconds']:.2f} seconds")
        print(f"Total Frames: {results['total_frames']}")
        print(f"FPS: {results['fps']:.2f}")
        print("\n" + "-"*60)
        print("ACTION COUNTS:")
        print("-"*60)
        for action_name, count in results['action_counts'].items():
            if action_name != 'no_action' and count > 0:
                print(f"  {action_name:20s}: {count:4d}")

        print("\n" + "-"*60)
        print("STATISTICS:")
        print("-"*60)
        stats = results['statistics']
        print(f"  Total Actions:     {results['total_actions']}")
        print(f"  Landed Hits:       {stats['landed_hits']}")
        print(f"  Blocks:            {stats['blocks']}")
        print(f"  Misses:            {stats['misses']}")
        print(f"  Hit Rate:          {stats['hit_rate']*100:.1f}%")
        print(f"  Block Rate:        {stats['block_rate']*100:.1f}%")
        print(f"  Miss Rate:         {stats['miss_rate']*100:.1f}%")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Run inference on video with ActionNet Transformer')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to input video file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results (optional)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Image size for model input')
    parser.add_argument('--num-frames', type=int, default=30,
                        help='Number of frames per sequence')
    parser.add_argument('--stride', type=int, default=15,
                        help='Frames to skip between sequences (default: 15 = 50%% overlap)')
    parser.add_argument('--min-peak-gap', type=int, default=5,
                        help='Minimum frames between peaks for peak detection')
    parser.add_argument('--min-peak-height', type=float, default=0.5,
                        help='Minimum confidence for peak detection')
    parser.add_argument('--min-action-duration', type=int, default=2,
                        help='Minimum frames for valid action')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Auto-detected if not specified')
    parser.add_argument('--boost-under-detected', action='store_true',
                        help='Apply confidence boosts to under-detected classes (body hits, blocks, miss_right)')

    args = parser.parse_args()

    class_boost = None
    if args.boost_under_detected:
        class_boost = {
            0: 1.0,
            1: 1.0,
            2: 1.5,
            3: 2.0,
            4: 1.3,
            5: 1.0,
            6: 1.0,
            7: 1.4,
        }
        logger.info("Applying confidence boosts to under-detected classes")
        logger.info(f"  body_hit_left: 1.5x, body_hit_right: 2.0x, block_left: 1.3x, miss_right: 1.4x")

    inference = VideoInference(
        checkpoint_path=args.checkpoint,
        img_size=args.img_size,
        num_frames=args.num_frames,
        device=args.device,
        min_peak_gap=args.min_peak_gap,
        min_peak_height=args.min_peak_height,
        min_action_duration=args.min_action_duration,
        class_confidence_boost=class_boost
    )

    results = inference.process_video(
        video_path=args.video,
        stride=args.stride,
        output_file=args.output
    )

    inference.print_summary(results)

    return results


if __name__ == '__main__':
    main()
