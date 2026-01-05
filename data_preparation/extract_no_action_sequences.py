"""
Extract "No Action" Sequences from Videos

This script:
1. Loads all converted CVAT annotations
2. Identifies "safe zones" in videos (frames at least 8 frames away from any annotation)
3. Extracts ~2500 "no action" sequences from these safe zones
4. Adds them to the training data with label 8 (no_action)
"""

import os
import json
import numpy as np
from pathlib import Path
import logging
import glob
import cv2
import random
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ACTIONS = [
    "head_hit_left", "head_hit_right",
    "body_hit_left", "body_hit_right",
    "block_left", "block_right",
    "miss_left", "miss_right",
    "no_action"
]

NO_ACTION_CLASS_ID = 8
MIN_DISTANCE_FROM_ACTION = 8
SEQUENCE_LENGTH = 30
TARGET_NO_ACTION_SEQUENCES = 2500


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


def get_video_frame_count(video_path):
    """Get total number of frames in video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def find_safe_zones(annotated_frames, total_frames, min_distance=8, sequence_length=30):
    """
    Find safe zones where we can extract no-action sequences.

    Args:
        annotated_frames: List of frame numbers with annotations
        total_frames: Total number of frames in video
        min_distance: Minimum frames away from any annotation
        sequence_length: Length of sequence to extract

    Returns:
        List of (start_frame, end_frame) tuples for safe zones
    """
    if not annotated_frames:

        if total_frames >= sequence_length:
            return [(0, total_frames - sequence_length)]
        return []


    annotated_frames = sorted(set(annotated_frames))

    safe_zones = []


    if annotated_frames[0] >= min_distance + sequence_length:
        safe_end = annotated_frames[0] - min_distance
        safe_zones.append((0, safe_end - sequence_length + 1))


    for i in range(len(annotated_frames) - 1):
        gap_start = annotated_frames[i] + min_distance
        gap_end = annotated_frames[i + 1] - min_distance

        if gap_end - gap_start >= sequence_length:
            safe_zones.append((gap_start, gap_end - sequence_length + 1))


    last_annotation = annotated_frames[-1]
    if total_frames - last_annotation >= min_distance + sequence_length:
        safe_start = last_annotation + min_distance
        safe_end = total_frames - sequence_length
        if safe_end >= safe_start:
            safe_zones.append((safe_start, safe_end))

    return safe_zones


def extract_no_action_sequences(
    converted_dir: str,
    video_base_dir: str,
    output_file: str,
    target_sequences: int = 2500,
    min_distance: int = 8,
    sequence_length: int = 30,
    random_seed: int = 42
):
    """
    Extract no-action sequences from videos.

    Args:
        converted_dir: Directory with converted annotation JSON files
        video_base_dir: Base directory containing video files
        output_file: Path to save no-action sequences JSON
        target_sequences: Target number of sequences to extract
        min_distance: Minimum frames away from any annotation
        sequence_length: Number of frames per sequence
        random_seed: Random seed for reproducibility
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    logger.info(f"Loading annotations from {converted_dir}")
    logger.info(f"Loading videos from {video_base_dir}")


    aqib_video_dir = "/home/aqibchoudhary/training_data/Olympic Boxing Punch Classification Video Dataset"
    if not os.path.exists(video_base_dir) and os.path.exists(aqib_video_dir):
        logger.info(f"Video dir not found at {video_base_dir}, trying {aqib_video_dir}")
        video_base_dir = aqib_video_dir


    converted_files = [f for f in os.listdir(converted_dir) if f.endswith('_converted.json')]
    logger.info(f"Found {len(converted_files)} annotation files")


    video_annotations = {}

    for task_file in converted_files:
        task_name = task_file.replace('_converted.json', '')

        annotation_path = os.path.join(converted_dir, task_file)
        try:
            with open(annotation_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load {annotation_path}: {e}")
            continue

        annotations = data.get('annotations', [])
        annotated_frames = []

        for ann in annotations:
            frame_num = ann.get('frame_number', 0)
            annotated_frames.append(frame_num)

        if annotated_frames:
            video_annotations[task_name] = annotated_frames

    logger.info(f"Found annotations for {len(video_annotations)} videos")


    video_safe_zones = {}

    for video_id, annotated_frames in tqdm(video_annotations.items(), desc="Finding safe zones"):
        video_path = find_video_file(video_id, video_base_dir)
        if not video_path:
            continue

        total_frames = get_video_frame_count(video_path)
        if total_frames is None:
            continue

        safe_zones = find_safe_zones(
            annotated_frames,
            total_frames,
            min_distance=min_distance,
            sequence_length=sequence_length
        )

        if safe_zones:
            video_safe_zones[video_id] = {
                'video_path': video_path,
                'safe_zones': safe_zones,
                'total_frames': total_frames
            }

    logger.info(f"Found safe zones in {len(video_safe_zones)} videos")


    all_sequence_options = []

    for video_id, info in video_safe_zones.items():
        for start, end in info['safe_zones']:

            for center_frame in range(start, end + 1, sequence_length // 2):
                if center_frame + sequence_length // 2 <= info['total_frames']:
                    all_sequence_options.append({
                        'video_id': video_id,
                        'video_path': info['video_path'],
                        'center_frame': center_frame
                    })

    logger.info(f"Found {len(all_sequence_options)} possible no-action sequences")


    if len(all_sequence_options) > target_sequences:
        selected_sequences = random.sample(all_sequence_options, target_sequences)
    else:
        selected_sequences = all_sequence_options
        logger.warning(f"Only {len(selected_sequences)} sequences available, less than target {target_sequences}")


    no_action_sequences = []
    for seq in selected_sequences:
        no_action_sequences.append({
            'video_id': seq['video_id'],
            'video_path': seq['video_path'],
            'frame_number': seq['center_frame'],
            'label': NO_ACTION_CLASS_ID,
            'punch_type': 'no_action'
        })


    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(no_action_sequences, f, indent=2)

    logger.info(f"Extracted {len(no_action_sequences)} no-action sequences")
    logger.info(f"Saved to {output_file}")

    return {
        'total_sequences': len(no_action_sequences),
        'videos_processed': len(video_safe_zones),
        'total_safe_zones': sum(len(info['safe_zones']) for info in video_safe_zones.values())
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Extract no-action sequences from videos')
    parser.add_argument('--converted-dir', required=True,
                        help='Directory with converted annotation JSON files')
    parser.add_argument('--video-dir', required=True,
                        help='Base directory containing video files')
    parser.add_argument('--output-file', default='no_action_sequences.json',
                        help='Output JSON file for no-action sequences')
    parser.add_argument('--target-sequences', type=int, default=2500,
                        help='Target number of no-action sequences to extract')
    parser.add_argument('--min-distance', type=int, default=8,
                        help='Minimum frames away from any annotation')
    parser.add_argument('--sequence-length', type=int, default=30,
                        help='Number of frames per sequence')
    args = parser.parse_args()

    stats = extract_no_action_sequences(
        args.converted_dir,
        args.video_dir,
        args.output_file,
        target_sequences=args.target_sequences,
        min_distance=args.min_distance,
        sequence_length=args.sequence_length
    )

    print(f"\n✓ No-action sequence extraction complete!")
    print(f"  Total sequences: {stats['total_sequences']}")
    print(f"  Videos processed: {stats['videos_processed']}")
    print(f"  Total safe zones: {stats['total_safe_zones']}")
