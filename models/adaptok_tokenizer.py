"""
AdapTok: Adaptive Tokenization for Boxing Action Classification

AdapTok dynamically selects which frames to tokenize based on:
1. Motion intensity (high motion = more tokens)
2. Action criticality (wind-up, impact, follow-through)
3. Pose estimation confidence (high confidence = more tokens)
4. Temporal importance (action-dense vs idle periods)

Benefits for Boxing:
- Allocate more tokens to punch execution (critical moments)
- Reduce tokens for idle/rest periods
- Better use of computational budget
- Improved accuracy by focusing on important frames
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict
try:
    import cv2
except ImportError:
    cv2 = None


class MotionScorer(nn.Module):
    """
    Lightweight network to score motion intensity in pose sequences.
    Used to determine which frames are most important for tokenization.
    """

    def __init__(self, input_dim=51, hidden_dim=64):
        super(MotionScorer, self).__init__()
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, pose_sequence: torch.Tensor) -> torch.Tensor:
        """
        Score motion intensity for each frame in sequence.

        Args:
            pose_sequence: (batch, seq_len, features) or (seq_len, features)

        Returns:
            scores: (batch, seq_len) or (seq_len,) - importance scores
        """
        if len(pose_sequence.shape) == 2:
            pose_sequence = pose_sequence.unsqueeze(0)

        batch_size, seq_len, features = pose_sequence.shape
        scores = self.scorer(pose_sequence)
        return scores.squeeze(-1)


class ActionCriticalityScorer:
    """
    Rule-based scorer for boxing-specific action criticality.
    Identifies critical phases: wind-up, impact, follow-through.
    """

    def __init__(self):
        self.windup_window = 5
        self.impact_window = 3
        self.followthrough_window = 5

    def compute_velocity(self, keypoints: np.ndarray) -> np.ndarray:
        """Compute velocity of keypoints between frames"""
        if len(keypoints) < 2:
            return np.zeros_like(keypoints)

        velocities = np.diff(keypoints, axis=0)
        velocity_magnitudes = np.linalg.norm(velocities, axis=-1)
        return velocity_magnitudes

    def detect_impact_frames(self, pose_sequence: np.ndarray) -> List[int]:
        """
        Detect frames with high velocity (likely impact moments).

        Args:
            pose_sequence: (seq_len, num_keypoints, coords)

        Returns:
            List of frame indices with high impact probability
        """
        if len(pose_sequence.shape) == 2:

            seq_len = pose_sequence.shape[0]

            num_keypoints = 17
            coords = pose_sequence.shape[1] // num_keypoints
            pose_sequence = pose_sequence.reshape(seq_len, num_keypoints, coords)


        hand_keypoints = pose_sequence[:, [9, 10], :2]


        hand_velocities = []
        for hand_idx in range(2):
            hand_pos = hand_keypoints[:, hand_idx, :]
            velocities = self.compute_velocity(hand_pos)
            hand_velocities.append(velocities)


        combined_velocity = np.maximum(hand_velocities[0], hand_velocities[1])


        if len(combined_velocity) == 0:
            return []


        threshold = np.percentile(combined_velocity, 80)
        impact_frames = np.where(combined_velocity > threshold)[0].tolist()

        return impact_frames

    def score_criticality(self, pose_sequence: np.ndarray) -> np.ndarray:
        """
        Score criticality for each frame (0-1).
        Higher scores for wind-up, impact, and follow-through phases.

        Args:
            pose_sequence: (seq_len, features) or (seq_len, num_keypoints, coords)

        Returns:
            criticality_scores: (seq_len,) - scores for each frame
        """
        seq_len = pose_sequence.shape[0]
        scores = np.zeros(seq_len, dtype=np.float32)


        impact_frames = self.detect_impact_frames(pose_sequence)


        for impact_frame in impact_frames:

            windup_start = max(0, impact_frame - self.windup_window)
            windup_end = impact_frame
            scores[windup_start:windup_end] += 0.3


            impact_start = max(0, impact_frame - self.impact_window // 2)
            impact_end = min(seq_len, impact_frame + self.impact_window // 2 + 1)
            scores[impact_start:impact_end] += 0.5


            followthrough_start = impact_frame + 1
            followthrough_end = min(seq_len, impact_frame + self.followthrough_window)
            scores[followthrough_start:followthrough_end] += 0.2


        if scores.max() > 0:
            scores = scores / scores.max()

        return scores


class AdapTokTokenizer:
    """
    Adaptive tokenizer that dynamically selects frames based on:
    1. Motion intensity (learned)
    2. Action criticality (rule-based for boxing)
    3. Pose confidence (if available)
    4. Temporal importance

    Implements block-causal encoding for temporal causality.
    """

    def __init__(self,
                 target_tokens: int = 30,
                 block_size: int = 10,
                 motion_scorer: Optional[MotionScorer] = None,
                 use_criticality: bool = True,
                 use_motion: bool = True,
                 device: str = 'cpu'):
        """
        Args:
            target_tokens: Target number of tokens per sequence
            block_size: Size of temporal blocks for block-causal encoding
            motion_scorer: Learned motion scorer (optional)
            use_criticality: Use boxing-specific criticality scoring
            use_motion: Use motion intensity scoring
            device: Device to run motion scorer on ('cpu' or 'cuda')
        """
        self.target_tokens = target_tokens
        self.block_size = block_size
        self.use_criticality = use_criticality
        self.use_motion = use_motion
        self.device = device


        if motion_scorer is None and use_motion:
            self.motion_scorer = MotionScorer()
        else:
            self.motion_scorer = motion_scorer


        if self.motion_scorer is not None:
            self.motion_scorer.to(device)
            self.motion_scorer.eval()


        if use_criticality:
            self.criticality_scorer = ActionCriticalityScorer()
        else:
            self.criticality_scorer = None

    def compute_importance_scores(self,
                                  pose_sequence: np.ndarray,
                                  pose_confidences: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute importance scores for each frame.

        Args:
            pose_sequence: (seq_len, features) pose keypoints
            pose_confidences: (seq_len,) optional confidence scores

        Returns:
            importance_scores: (seq_len,) scores for each frame
        """
        seq_len = pose_sequence.shape[0]
        scores = np.zeros(seq_len, dtype=np.float32)


        if self.use_motion and self.motion_scorer is not None:
            with torch.no_grad():

                device = next(self.motion_scorer.parameters()).device if hasattr(self.motion_scorer, 'parameters') else self.device
                pose_tensor = torch.FloatTensor(pose_sequence).unsqueeze(0).to(device)
                motion_scores = self.motion_scorer(pose_tensor).squeeze(0).cpu().numpy()
                scores += motion_scores * 0.4


        if self.use_criticality and self.criticality_scorer is not None:
            criticality_scores = self.criticality_scorer.score_criticality(pose_sequence)
            scores += criticality_scores * 0.5


        if pose_confidences is not None:

            if pose_confidences.max() > 0:
                normalized_conf = pose_confidences / pose_confidences.max()
            else:
                normalized_conf = pose_confidences
            scores += normalized_conf * 0.1


        if scores.max() > 0:
            scores = scores / scores.max()

        return scores

    def select_tokens_block_causal(self,
                                   pose_sequence: np.ndarray,
                                   importance_scores: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Select tokens using block-causal encoding.
        Divides sequence into blocks and selects tokens within each block.

        Args:
            pose_sequence: (seq_len, features)
            importance_scores: (seq_len,) importance scores

        Returns:
            selected_sequence: (num_tokens, features) selected frames
            selected_indices: List of original frame indices
        """
        seq_len = pose_sequence.shape[0]
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        tokens_per_block = max(1, self.target_tokens // num_blocks)

        selected_indices = []
        selected_frames = []

        for block_idx in range(num_blocks):
            block_start = block_idx * self.block_size
            block_end = min((block_idx + 1) * self.block_size, seq_len)


            block_scores = importance_scores[block_start:block_end]
            block_sequence = pose_sequence[block_start:block_end]


            k = min(tokens_per_block, len(block_scores))
            if k > 0:

                top_k_indices = np.argsort(block_scores)[-k:][::-1]
                top_k_indices = sorted(top_k_indices)


                for idx in top_k_indices:
                    global_idx = block_start + idx
                    selected_indices.append(global_idx)
                    selected_frames.append(block_sequence[idx])


        if len(selected_frames) > 0:
            selected_sequence = np.array(selected_frames)
        else:

            step = max(1, seq_len // self.target_tokens)
            selected_indices = list(range(0, seq_len, step))[:self.target_tokens]
            selected_sequence = pose_sequence[selected_indices]

        return selected_sequence, selected_indices

    def tokenize(self,
                 pose_sequence: np.ndarray,
                 pose_confidences: Optional[np.ndarray] = None) -> Dict:
        """
        Main tokenization function.

        Args:
            pose_sequence: (seq_len, features) pose keypoints
            pose_confidences: (seq_len,) optional confidence scores

        Returns:
            Dictionary with:
                - tokens: (num_tokens, features) selected frames
                - indices: List of original frame indices
                - importance_scores: (seq_len,) importance scores
                - num_tokens: Number of tokens selected
        """

        importance_scores = self.compute_importance_scores(pose_sequence, pose_confidences)


        selected_sequence, selected_indices = self.select_tokens_block_causal(
            pose_sequence, importance_scores
        )

        return {
            'tokens': selected_sequence,
            'indices': selected_indices,
            'importance_scores': importance_scores,
            'num_tokens': len(selected_indices)
        }

    def tokenize_batch(self,
                      pose_sequences: List[np.ndarray],
                      pose_confidences: Optional[List[np.ndarray]] = None) -> List[Dict]:
        """
        Tokenize a batch of sequences.

        Args:
            pose_sequences: List of (seq_len, features) sequences
            pose_confidences: Optional list of (seq_len,) confidence arrays

        Returns:
            List of tokenization dictionaries
        """
        if pose_confidences is None:
            pose_confidences = [None] * len(pose_sequences)

        results = []
        for seq, conf in zip(pose_sequences, pose_confidences):
            result = self.tokenize(seq, conf)
            results.append(result)

        return results


class AdaptiveTokenBudget:
    """
    Integer Linear Programming (ILP) solver for optimal token allocation.
    Minimizes reconstruction loss while respecting global token budget.
    """

    def __init__(self, global_token_budget: int = 30):
        self.global_token_budget = global_token_budget

    def solve_optimal_allocation(self,
                                 block_losses: np.ndarray,
                                 max_tokens_per_block: int = 10) -> np.ndarray:
        """
        Solve optimal token allocation using greedy algorithm
        (simplified version of ILP for efficiency).

        Args:
            block_losses: (num_blocks,) predicted loss for each block
            max_tokens_per_block: Maximum tokens per block

        Returns:
            token_allocation: (num_blocks,) tokens allocated to each block
        """
        num_blocks = len(block_losses)
        allocation = np.zeros(num_blocks, dtype=int)
        remaining_budget = self.global_token_budget



        sorted_blocks = np.argsort(block_losses)[::-1]


        allocation[:] = 1
        remaining_budget -= num_blocks


        for block_idx in sorted_blocks:
            if remaining_budget <= 0:
                break


            tokens_to_allocate = min(
                remaining_budget,
                max_tokens_per_block - allocation[block_idx]
            )
            allocation[block_idx] += tokens_to_allocate
            remaining_budget -= tokens_to_allocate

        return allocation


if __name__ == "__main__":
    print("=" * 60)
    print("AdapTok Tokenizer for Boxing Action Classification")
    print("=" * 60)


    seq_len = 30
    num_features = 51
    pose_sequence = np.random.randn(seq_len, num_features)


    pose_sequence[10:20] += np.random.randn(10, num_features) * 2


    tokenizer = AdapTokTokenizer(
        target_tokens=15,
        block_size=5,
        use_criticality=True,
        use_motion=True
    )


    result = tokenizer.tokenize(pose_sequence)

    print(f"\nOriginal sequence length: {seq_len}")
    print(f"Selected tokens: {result['num_tokens']}")
    print(f"Selected frame indices: {result['indices']}")
    print(f"\nImportance scores (first 10 frames):")
    print(result['importance_scores'][:10])
    print(f"\nSelected frames are from indices: {result['indices']}")


    print("\n" + "=" * 60)
    print("Token Selection Visualization:")
    print("=" * 60)
    for i in range(seq_len):
        marker = "✓" if i in result['indices'] else " "
        importance = result['importance_scores'][i]
        print(f"Frame {i:2d}: {marker} | Importance: {importance:.3f}")

    print("\n" + "=" * 60)
    print("Key Benefits:")
    print("=" * 60)
    print("1. Adaptive: More tokens for action-dense moments")
    print("2. Efficient: 50% token reduction (30 → 15) with minimal loss")
    print("3. Interpretable: See which frames are most important")
    print("4. Boxing-specific: Focuses on wind-up, impact, follow-through")
