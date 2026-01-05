"""
Prepare Training Data from Raw Video Frames

This script:
1. Loads converted CVAT annotations
2. Extracts raw video frame sequences (no pose estimation)
3. Creates training sequences (30 frames each)
4. Splits data into 80% training and 20% validation
5. Saves train.json and val.json with video paths and frame numbers
"""

import os
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ACTIONS = [
    "head_hit_left", "head_hit_right",
    "body_hit_left", "body_hit_right",
    "block_left", "block_right",
    "miss_left", "miss_right",
    "no_action"
]


def find_video_file(video_id, video_base_dir):
    """Find video file for a given video ID"""
    patterns = [
        f"{video_base_dir}/{video_id}/data/*.mp4",
        f"{video_base_dir}/{video_id}/data/*.MOV",
        f"{video_base_dir}/{video_id}/data/*.avi",
        f"{video_base_dir}/*/{video_id}/data/*.mp4",
        f"{video_base_dir}/*/{video_id}/data/*.MOV",
        f"/home/aqibchoudhary/training_data/Olympic Boxing Punch Classification Video Dataset/{video_id}/data/*.mp4",
        f"/home/aqibchoudhary/training_data/Olympic Boxing Punch Classification Video Dataset/{video_id}/data/*.MOV",
    ]

    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    return None


def prepare_video_training_data(
    converted_dir: str,
    video_base_dir: str,
    output_dir: str = "training_data_video",
    train_split: float = 0.8,
    sequence_length: int = 30,
    random_state: int = 42,
    no_action_file: str = None
):
    """
    Prepare training data from raw video frames

    Args:
        converted_dir: Directory with converted annotation JSON files
        video_base_dir: Base directory containing video files
        output_dir: Output directory for train.json and val.json
        train_split: Train/val split ratio
        sequence_length: Number of frames per sequence
        random_state: Random seed
    """
    os.makedirs(output_dir, exist_ok=True)

    training_sequences = []
    class_counts = {i: 0 for i in range(len(ACTIONS))}
    NO_ACTION_CLASS_ID = 8

    logger.info(f"Loading annotations from {converted_dir}")
    logger.info(f"Loading videos from {video_base_dir}")


    aqib_video_dir = "/home/aqibchoudhary/training_data/Olympic Boxing Punch Classification Video Dataset"
    if not os.path.exists(video_base_dir) and os.path.exists(aqib_video_dir):
        logger.info(f"Video dir not found at {video_base_dir}, trying {aqib_video_dir}")
        video_base_dir = aqib_video_dir


    converted_files = [f for f in os.listdir(converted_dir) if f.endswith('_converted.json')]
    logger.info(f"Found {len(converted_files)} annotation files")

    processed = 0
    skipped = 0

    for task_file in converted_files:
        task_name = task_file.replace('_converted.json', '')


        annotation_path = os.path.join(converted_dir, task_file)
        try:
            with open(annotation_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {annotation_path}: {e}")
            continue


        video_path = find_video_file(task_name, video_base_dir)
        if not video_path:
            logger.warning(f"Video not found for {task_name}, skipping")
            skipped += 1
            continue


        annotations = data.get('annotations', [])
        if not annotations:
            logger.warning(f"No annotations in {task_file}")
            continue

        for ann in annotations:
            frame_num = ann.get('frame_number', 0)
            class_id = ann.get('class_id', -1)

            if class_id < 0 or class_id >= len(ACTIONS):
                continue

            training_sequences.append({
                'video_id': task_name,
                'video_path': video_path,
                'frame_number': frame_num,
                'label': class_id,
                'punch_type': ann.get('punch_type', ACTIONS[class_id])
            })

            class_counts[class_id] += 1

        processed += 1

    logger.info(f"Processed {processed} videos, skipped {skipped} videos")


    if no_action_file and os.path.exists(no_action_file):
        logger.info(f"Loading no-action sequences from {no_action_file}")
        try:
            with open(no_action_file, 'r') as f:
                no_action_sequences = json.load(f)


            for seq in no_action_sequences:
                training_sequences.append({
                    'video_id': seq['video_id'],
                    'video_path': seq.get('video_path', find_video_file(seq['video_id'], video_base_dir)),
                    'frame_number': seq['frame_number'],
                    'label': NO_ACTION_CLASS_ID,
                    'punch_type': 'no_action'
                })
                class_counts[NO_ACTION_CLASS_ID] += 1

            logger.info(f"Added {len(no_action_sequences)} no-action sequences")
        except Exception as e:
            logger.warning(f"Failed to load no-action sequences: {e}")

    if not training_sequences:
        raise ValueError("No training sequences created!")

    logger.info(f"Created {len(training_sequences)} training sequences")
    logger.info("Class distribution:")
    for i, action in enumerate(ACTIONS):
        logger.info(f"  {action}: {class_counts[i]}")


    labels = [seq['label'] for seq in training_sequences]
    train_data, val_data = train_test_split(
        training_sequences,
        test_size=1-train_split,
        random_state=random_state,
        stratify=labels
    )


    train_file = os.path.join(output_dir, 'train.json')
    val_file = os.path.join(output_dir, 'val.json')

    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(val_file, 'w') as f:
        json.dump(val_data, f, indent=2)

    logger.info(f"Saved {len(train_data)} training sequences to {train_file}")
    logger.info(f"Saved {len(val_data)} validation sequences to {val_file}")

    return {
        'total_sequences': len(training_sequences),
        'train_sequences': len(train_data),
        'val_sequences': len(val_data),
        'processed_videos': processed,
        'skipped_videos': skipped,
        'class_distribution': class_counts
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--converted-dir', required=True,
                        help='Directory with converted annotation JSON files')
    parser.add_argument('--video-dir', required=True,
                        help='Base directory containing video files')
    parser.add_argument('--output-dir', default='training_data_video',
                        help='Output directory for train.json and val.json')
    parser.add_argument('--train-split', type=float, default=0.8)
    parser.add_argument('--no-action-file', type=str, default=None,
                        help='Path to no_action_sequences.json (optional)')
    args = parser.parse_args()

    stats = prepare_video_training_data(
        args.converted_dir,
        args.video_dir,
        args.output_dir,
        args.train_split,
        no_action_file=args.no_action_file
    )
    print(f"\n✓ Training data preparation complete!")
    print(f"  Total sequences: {stats['total_sequences']}")
    print(f"  Train: {stats['train_sequences']}, Val: {stats['val_sequences']}")
