"""
Pre-extract video frames for faster training

This script extracts frame sequences from videos and saves them as numpy arrays.
This eliminates video decoding overhead during training, providing 4-10x speedup.
"""

import os
import json
import numpy as np
import cv2
import glob
from pathlib import Path
from tqdm import tqdm
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_video_file(video_id, video_base_dir):
    """Find video file for a given video ID"""
    patterns = [
        f"{video_base_dir}/*/{video_id}/data/*.mp4",
        f"{video_base_dir}/*/{video_id}/data/*.MOV",
        f"{video_base_dir}/*/{video_id}/data/*.avi",
        f"{video_base_dir}/{video_id}/data/*.mp4",
        f"{video_base_dir}/{video_id}/data/*.MOV",
        f"{video_base_dir}/{video_id}/data/*.avi",
    ]

    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


def extract_frame_sequence(video_path, center_frame, num_frames=30, img_size=224):
    """Extract sequence of frames around center_frame"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, center_frame - num_frames // 2)
    end_frame = min(total_frames, center_frame + num_frames // 2)

    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (img_size, img_size))
        frames.append(frame_resized)

    cap.release()

    while len(frames) < num_frames:
        if frames:
            frames.append(frames[-1])
        else:
            frames.append(np.zeros((img_size, img_size, 3), dtype=np.uint8))

    frames = frames[:num_frames]

    return np.array(frames, dtype=np.uint8)


def pre_extract_frames(json_file, video_base_dir, output_dir, img_size=224, num_frames=30):
    """
    Pre-extract all frame sequences from training data

    Args:
        json_file: Path to train.json or val.json
        video_base_dir: Base directory containing video files
        output_dir: Directory to save extracted frames
        img_size: Target image size
        num_frames: Number of frames per sequence
    """

    with open(json_file, 'r') as f:
        data = json.load(f)

    logger.info(f"Processing {len(data)} sequences from {json_file}")


    os.makedirs(output_dir, exist_ok=True)


    successful = 0
    failed = 0
    failed_items = []


    for idx, item in enumerate(tqdm(data, desc="Extracting frames")):
        video_id = item['video_id']
        frame_number = item['frame_number']
        label = item['label']


        video_path = find_video_file(video_id, video_base_dir)
        if video_path is None:
            logger.warning(f"Video not found for {video_id}, skipping")
            failed += 1
            failed_items.append(item)
            continue

        try:


            existing_files = glob.glob(
                os.path.join(output_dir, f"*_{video_id}_{frame_number:06d}_{label}.npy")
            )

            if existing_files:

                successful += 1
                continue


            frames = extract_frame_sequence(
                video_path,
                frame_number,
                num_frames=num_frames,
                img_size=img_size
            )



            filename = f"{idx:06d}_{video_id}_{frame_number:06d}_{label}.npy"
            filepath = os.path.join(output_dir, filename)

            np.save(filepath, frames)
            successful += 1

        except Exception as e:
            logger.error(f"Failed to extract frames for {video_id} frame {frame_number}: {e}")
            failed += 1
            failed_items.append(item)
            continue


    metadata = {
        'total_items': len(data),
        'successful': successful,
        'failed': failed,
        'img_size': img_size,
        'num_frames': num_frames,
        'failed_items': failed_items
    }

    metadata_path = os.path.join(output_dir, 'metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nExtraction complete!")
    logger.info(f"  Successful: {successful}/{len(data)}")
    logger.info(f"  Failed: {failed}/{len(data)}")
    logger.info(f"  Output directory: {output_dir}")

    return successful, failed


def main():
    parser = argparse.ArgumentParser(description='Pre-extract video frames for faster training')
    parser.add_argument('--train-file', type=str, required=True,
                        help='Path to train.json')
    parser.add_argument('--val-file', type=str, required=True,
                        help='Path to val.json')
    parser.add_argument('--video-dir', type=str, required=True,
                        help='Base directory containing video files')
    parser.add_argument('--output-dir', type=str, default='extracted_frames',
                        help='Directory to save extracted frames')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Target image size')
    parser.add_argument('--num-frames', type=int, default=30,
                        help='Number of frames per sequence')

    args = parser.parse_args()


    train_output = os.path.join(args.output_dir, 'train')
    val_output = os.path.join(args.output_dir, 'val')

    logger.info("=" * 60)
    logger.info("Pre-extracting Training Frames")
    logger.info("=" * 60)
    train_success, train_failed = pre_extract_frames(
        args.train_file,
        args.video_dir,
        train_output,
        img_size=args.img_size,
        num_frames=args.num_frames
    )

    logger.info("\n" + "=" * 60)
    logger.info("Pre-extracting Validation Frames")
    logger.info("=" * 60)
    val_success, val_failed = pre_extract_frames(
        args.val_file,
        args.video_dir,
        val_output,
        img_size=args.img_size,
        num_frames=args.num_frames
    )

    logger.info("\n" + "=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Training: {train_success} successful, {train_failed} failed")
    logger.info(f"Validation: {val_success} successful, {val_failed} failed")
    logger.info(f"Total: {train_success + val_success} successful, {train_failed + val_failed} failed")
    logger.info(f"\nFrames saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
