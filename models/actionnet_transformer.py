"""
Transformer-based ActionNet for Punch Classification

This is a reference implementation showing how ActionNet could be replaced
with a transformer architecture for better performance and interpretability.

Benefits over LSTM:
1. Long-range temporal dependencies
2. Attention visualization
3. Better handling of variable-length sequences
4. Multi-scale temporal understanding
5. Robustness to pose estimation errors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Add positional encoding to input sequences"""

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class ActionNetTransformer(nn.Module):
    """
    Transformer-based ActionNet for punch classification

    Architecture:
    - Input embedding layer
    - Positional encoding
    - Transformer encoder (multi-head attention)
    - Classification head

    Advantages over LSTM:
    1. Parallel processing of all frames
    2. Attention mechanism for interpretability
    3. Better long-range dependencies
    4. Multi-scale temporal understanding
    """

    def __init__(self, input_size=51, hidden_size=128, num_layers=3,
                 num_heads=8, num_actions=8, dropout=0.1, max_seq_len=100):
        super(ActionNetTransformer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_actions = num_actions


        self.input_projection = nn.Linear(input_size, hidden_size)


        self.pos_encoder = PositionalEncoding(hidden_size, max_seq_len)


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



        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_actions)
        )


        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            return_attention: If True, return attention weights for visualization

        Returns:
            output: Classification logits (batch, num_actions)
            attention_weights: (optional) Attention weights for visualization
        """
        batch_size, seq_len, _ = x.shape


        x = self.input_projection(x)
        x = self.dropout(x)


        cls_tokens = self.cls_token.expand(1, batch_size, -1)
        x = x.transpose(0, 1)
        x = torch.cat([cls_tokens, x], dim=0)


        x = self.pos_encoder(x)


        if return_attention:

            attention_weights = []
            for layer in self.transformer_encoder.layers:
                x, attn = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=False)
                attention_weights.append(attn)
            encoded = x
        else:
            encoded = self.transformer_encoder(x)


        cls_output = encoded[0]


        output = self.classifier(cls_output)

        if return_attention:
            return output, attention_weights
        return output

    def get_attention_maps(self, x):
        """
        Get attention maps for visualization

        Returns attention weights showing which frames the model focuses on
        """
        _, attention_weights = self.forward(x, return_attention=True)
        return attention_weights


class ActionNetTransformerPooled(nn.Module):
    """
    Alternative transformer architecture using pooling instead of CLS token

    This version pools all frame representations instead of using a CLS token.
    Can be more interpretable as it shows which frames contribute to classification.
    """

    def __init__(self, input_size=51, hidden_size=128, num_layers=3,
                 num_heads=8, num_actions=8, dropout=0.1, max_seq_len=100):
        super(ActionNetTransformerPooled, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size


        self.input_projection = nn.Linear(input_size, hidden_size)


        self.pos_encoder = PositionalEncoding(hidden_size, max_seq_len)


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


        self.attention_pool = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=1,
            batch_first=False
        )


        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_actions)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        """
        Forward pass with attention-based pooling

        Returns pooled representation and attention weights showing frame importance
        """
        batch_size, seq_len, _ = x.shape


        x = self.input_projection(x)
        x = self.dropout(x)
        x = x.transpose(0, 1)


        x = self.pos_encoder(x)


        encoded = self.transformer_encoder(x)



        query = encoded.mean(dim=0, keepdim=True)
        pooled, attention_weights = self.attention_pool(
            query, encoded, encoded, need_weights=return_attention
        )
        pooled = pooled.squeeze(0)


        output = self.classifier(pooled)

        if return_attention:
            return output, attention_weights
        return output


if __name__ == "__main__":

    batch_size = 2
    seq_len = 30
    input_size = 51
    x = torch.randn(batch_size, seq_len, input_size)

    print("=" * 60)
    print("Transformer-based ActionNet Comparison")
    print("=" * 60)


    from actionnet import ActionNet
    lstm_model = ActionNet(hidden_size=64, input_size=input_size, num_actions=8)
    lstm_output = lstm_model(x)
    print(f"\nLSTM Model:")
    print(f"  Output shape: {lstm_output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")


    transformer_model = ActionNetTransformer(
        input_size=input_size,
        hidden_size=128,
        num_layers=3,
        num_heads=8,
        num_actions=8
    )
    transformer_output, attention = transformer_model(x, return_attention=True)
    print(f"\nTransformer Model (CLS token):")
    print(f"  Output shape: {transformer_output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")
    print(f"  Attention layers: {len(attention)}")
    print(f"  Attention shape per layer: {attention[0].shape}")


    transformer_pooled = ActionNetTransformerPooled(
        input_size=input_size,
        hidden_size=128,
        num_layers=3,
        num_heads=8,
        num_actions=8
    )
    pooled_output, pooled_attention = transformer_pooled(x, return_attention=True)
    print(f"\nTransformer Model (Pooled):")
    print(f"  Output shape: {pooled_output.shape}")
    print(f"  Parameters: {sum(p.numel() for p in transformer_pooled.parameters()):,}")
    print(f"  Pooling attention shape: {pooled_attention.shape}")

    print("\n" + "=" * 60)
    print("Key Advantages of Transformer:")
    print("=" * 60)
    print("1. Attention visualization - see which frames matter")
    print("2. Parallel processing - faster on modern GPUs")
    print("3. Long-range dependencies - better combo detection")
    print("4. Multi-scale understanding - micro + macro patterns")
    print("5. Robustness - can ignore noisy frames via attention")
