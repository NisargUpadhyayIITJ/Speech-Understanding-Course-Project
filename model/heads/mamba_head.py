from __future__ import annotations

import torch
import torch.nn as nn
from transformers import MambaConfig, MambaModel

from ..kan import KANLinear


class MambaHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        state_dim: int = 16,
        conv_kernel: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.bridge = KANLinear(input_dim, hidden_dim)
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.mamba = MambaModel(
            MambaConfig(
                hidden_size=hidden_dim,
                state_size=state_dim,
                num_hidden_layers=num_layers,
                intermediate_size=hidden_dim * expand,
                conv_kernel=conv_kernel,
                vocab_size=8,
            )
        )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2
        self.classifier = KANLinear(self.output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.bridge(x)
        x = self.pre_norm(x)
        x = self.mamba(inputs_embeds=x, use_cache=False).last_hidden_state
        pooled_mean = x.mean(dim=1)
        pooled_max = x.max(dim=1).values
        embedding = torch.cat([pooled_mean, pooled_max], dim=1)
        logits = self.classifier(self.dropout(embedding))
        return logits, embedding
