"""
ActionNet Transformer for Raw Video Frames

Processes raw video frames directly without pose estimation.
Uses Vision Transformer (ViT) approach with frame patches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple

from actionnet_transformer import PositionalEncoding


class FramePatchEmbedding(nn.Module):
    """Convert video frames into patch embeddings"""

    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, channels, height, width) or (batch, channels, height, width)
        Returns:
            patches: (batch, seq_len, num_patches, embed_dim) or (batch, num_patches, embed_dim)
        """
        if len(x.shape) == 4:

            x = x.unsqueeze(1)

        batch_size, seq_len, channels, height, width = x.shape


        x = x.view(batch_size * seq_len, channels, height, width)


        patches = self.patch_embed(x)


        patches = patches.flatten(2).transpose(1, 2)


        patches = patches.view(batch_size, seq_len, self.num_patches, self.embed_dim)


        patches = self.norm(patches)

        return patches


class ActionNetTransformerVideo(nn.Module):
    """
    Transformer for raw video frames.

    Architecture:
    1. Frame patches: Each frame → patches → embeddings
    2. Temporal sequence: Sequence of frame embeddings
    3. Transformer encoder: Processes temporal sequence
    4. Classification head: Outputs action class
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 num_frames=30,
                 hidden_size=128,
                 num_layers=3,
                 num_heads=8,
                 num_actions=9,
                 dropout=0.1,
                 in_channels=3):
        super(ActionNetTransformerVideo, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.hidden_size = hidden_size
        self.num_actions = num_actions


        self.frame_embedder = FramePatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=hidden_size
        )

        num_patches = (img_size // patch_size) ** 2


        self.frame_projection = nn.Linear(num_patches * hidden_size, hidden_size)


        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))


        self.pos_encoder = PositionalEncoding(hidden_size, max_len=num_frames + 1)


        self.prediction_mode = 'dense'


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )


        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_actions)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False, prediction_mode=None):
        """
        Args:
            x: (batch, seq_len, channels, height, width) - raw video frames
            return_attention: Return attention weights
            prediction_mode: 'dense' for per-frame predictions, 'cls' for sequence-level
        Returns:
            Dictionary with output logits and optional attention
            - 'dense' mode: output shape (batch, seq_len, num_actions)
            - 'cls' mode: output shape (batch, num_actions)
        """
        if prediction_mode is None:
            prediction_mode = self.prediction_mode

        batch_size, seq_len = x.shape[0], x.shape[1]


        patch_embeds = self.frame_embedder(x)


        frame_embeds = patch_embeds.view(batch_size, seq_len, -1)
        frame_embeds = self.frame_projection(frame_embeds)


        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat([cls_tokens, frame_embeds], dim=1)


        sequence = sequence.transpose(0, 1)
        sequence = self.pos_encoder(sequence)


        encoded = self.transformer_encoder(sequence)

        if prediction_mode == 'dense':

            frame_outputs = encoded[1:]
            frame_outputs = frame_outputs.transpose(0, 1).contiguous()



            batch_seq_len, hidden = frame_outputs.shape[0] * frame_outputs.shape[1], frame_outputs.shape[2]
            frame_outputs_flat = frame_outputs.reshape(batch_seq_len, hidden)
            frame_predictions_flat = self.classifier(frame_outputs_flat)
            output = frame_predictions_flat.reshape(batch_size, seq_len, self.num_actions)
        else:

            cls_output = encoded[0]
            output = self.classifier(cls_output)

        result = {'output': output, 'prediction_mode': prediction_mode}

        if return_attention:

            result['attention_weights'] = None

        return result
