"""GRU text classifier. Input: LongTensor (B, L) of token ids, padded with `pad_idx`."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class GRUClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lengths = (x != self.pad_idx).sum(dim=1).clamp(min=1).cpu()
        emb = self.drop(self.embed(x))
        packed = pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)  # h: (num_layers, B, hidden)
        return self.fc(self.drop(h[-1]))
