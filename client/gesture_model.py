"""Transformer encoder model for gesture sequence classification.

Architecture:
    Linear projection: input_dim (260) → d_model (128)
    Learnable positional encoding (max_seq_len positions)
    Transformer encoder: 3 layers, 4 heads, dim_feedforward=256, dropout=0.3
    Classification head: masked mean pooling → Linear → num_classes
"""

import torch
import torch.nn as nn


class GestureTransformer(nn.Module):
    """Small Transformer encoder for classifying variable-length pose+hand sequences.

    Args:
        input_dim: Feature dimension per frame (default 260).
        num_classes: Number of gesture classes.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of Transformer encoder layers.
        dim_feedforward: Feedforward network dimension.
        dropout: Dropout rate.
        max_seq_len: Maximum sequence length (for positional encoding).
    """

    def __init__(
        self,
        input_dim: int = 260,
        num_classes: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        max_seq_len: int = 120,
    ):
        super().__init__()
        self.d_model = d_model

        # Project input features to model dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable positional encoding
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim) — input features.
            mask: (batch, seq_len) — 1.0 for real frames, 0.0 for padding.

        Returns:
            logits: (batch, num_classes)
        """
        B, T, _ = x.shape

        # Project to d_model
        x = self.input_proj(x)  # (B, T, d_model)

        # Add positional encoding
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        x = x + self.pos_embedding(positions)  # (B, T, d_model)

        x = self.dropout(x)

        # Transformer expects src_key_padding_mask: True = ignore (padded)
        padding_mask = mask == 0  # (B, T), True where padded

        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)  # (B, T, d_model)

        # Masked mean pooling — average only non-padded positions
        mask_expanded = mask.unsqueeze(-1)  # (B, T, 1)
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)  # (B, d_model)

        logits = self.classifier(x)  # (B, num_classes)
        return logits
