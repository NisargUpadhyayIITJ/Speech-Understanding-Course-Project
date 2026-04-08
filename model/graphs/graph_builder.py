from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..kan import KANLinear


class DynamicGraphBuilder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        mode: str = "fixed",
        edge_score: str = "dot",
        topk: int = 8,
        symmetric: bool = True,
        include_self_loops: bool = True,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.edge_score = edge_score
        self.topk = topk
        self.symmetric = symmetric
        self.include_self_loops = include_self_loops

        if edge_score == "mlp":
            self.edge_mlp = nn.Sequential(
                KANLinear(hidden_dim * 2, hidden_dim),
                nn.Tanh(),
                KANLinear(hidden_dim, 1),
            )
        else:
            self.edge_mlp = None
        self.last_stats: dict[str, float] = {}

    def _compute_stats(self, adjacency: torch.Tensor) -> dict[str, float]:
        batch_size, num_nodes, _ = adjacency.shape
        degrees = adjacency.sum(dim=-1)
        edge_density = adjacency.mean()

        if self.include_self_loops:
            eye = torch.eye(num_nodes, device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0)
            off_diagonal = adjacency * (1.0 - eye)
            off_diagonal_density = off_diagonal.sum() / max(batch_size * num_nodes * max(num_nodes - 1, 1), 1)
        else:
            off_diagonal_density = edge_density

        return {
            "num_nodes": float(num_nodes),
            "avg_degree": float(degrees.mean().detach().cpu()),
            "max_degree": float(degrees.max().detach().cpu()),
            "edge_density": float(edge_density.detach().cpu()),
            "offdiag_edge_density": float(off_diagonal_density.detach().cpu()),
        }

    def _pairwise_scores(self, x: torch.Tensor) -> torch.Tensor:
        if self.edge_score == "dot":
            normalized = F.normalize(x, dim=-1)
            return torch.matmul(normalized, normalized.transpose(1, 2))

        if self.edge_score != "mlp":
            raise ValueError(f"Unknown edge score mode: {self.edge_score}")

        batch_size, num_nodes, hidden_dim = x.shape
        left = x.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, hidden_dim)
        right = x.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, hidden_dim)
        pairwise = torch.cat([left, right], dim=-1)
        return self.edge_mlp(pairwise).squeeze(-1)

    def _apply_topk(self, scores: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = scores.shape
        if self.topk <= 0 or self.topk >= num_nodes:
            adjacency = torch.ones_like(scores)
        else:
            indices = torch.topk(scores, k=self.topk, dim=-1).indices
            adjacency = torch.zeros_like(scores)
            adjacency.scatter_(-1, indices, 1.0)

        if self.symmetric:
            adjacency = torch.maximum(adjacency, adjacency.transpose(1, 2))

        if self.include_self_loops:
            eye = torch.eye(num_nodes, device=scores.device).unsqueeze(0).expand(batch_size, -1, -1)
            adjacency = torch.maximum(adjacency, eye)
        return adjacency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "fixed":
            batch_size, num_nodes, _ = x.shape
            adjacency = torch.ones(batch_size, num_nodes, num_nodes, device=x.device, dtype=x.dtype)
            if not self.include_self_loops:
                eye = torch.eye(num_nodes, device=x.device).unsqueeze(0)
                adjacency = adjacency - eye
            self.last_stats = self._compute_stats(adjacency)
            return adjacency

        if self.mode != "learned":
            raise ValueError(f"Unknown graph mode: {self.mode}")

        scores = self._pairwise_scores(x)
        adjacency = self._apply_topk(scores).to(x.dtype)
        self.last_stats = self._compute_stats(adjacency)
        return adjacency


class MaskedGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.query = KANLinear(in_dim, out_dim)
        self.key = KANLinear(in_dim, out_dim)
        self.value = KANLinear(in_dim, out_dim)
        self.out_proj = KANLinear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(query.size(-1))
        scores = scores.masked_fill(adjacency <= 0, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        out = torch.matmul(attention, value)
        out = self.out_proj(out)
        return self.norm(F.gelu(out))
