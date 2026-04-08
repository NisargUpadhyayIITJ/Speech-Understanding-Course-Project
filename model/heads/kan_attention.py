from __future__ import annotations

import torch
import torch.nn as nn

from ..kan import KANLinear


class KANFeedForward(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: int = 4, dropout: float = 0.1, use_kan_feedforward: bool = True) -> None:
        super().__init__()
        inner_dim = hidden_dim * mlp_ratio
        linear_cls = KANLinear if use_kan_feedforward else nn.Linear
        self.ff1 = linear_cls(hidden_dim, inner_dim)
        self.ff2 = linear_cls(inner_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.ff2(x)
        return self.dropout(x)


class KANAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int, dropout: float, use_kan_feedforward: bool) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.feedforward = KANFeedForward(hidden_dim, mlp_ratio=mlp_ratio, dropout=dropout, use_kan_feedforward=use_kan_feedforward)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.dropout(attn_output)
        x = x + self.feedforward(self.norm2(x))
        return x


class KANAttentionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        num_classes: int = 2,
        use_kan_feedforward: bool = True,
    ) -> None:
        super().__init__()
        self.bridge = KANLinear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                KANAttentionBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_kan_feedforward=use_kan_feedforward,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2
        self.classifier = KANLinear(self.output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.bridge(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        pooled_mean = x.mean(dim=1)
        pooled_max = x.max(dim=1).values
        embedding = torch.cat([pooled_mean, pooled_max], dim=1)
        logits = self.classifier(self.dropout(embedding))
        return logits, embedding

