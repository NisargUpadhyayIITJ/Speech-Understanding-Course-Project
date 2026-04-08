from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn

from ..kan import KANLinear
from ..pool import GraphPool


def _aggregate_neighbors(x: torch.Tensor, adjacency: torch.Tensor, aggregator: str) -> torch.Tensor:
    adjacency = adjacency.to(x.dtype)
    degree = adjacency.sum(dim=-1, keepdim=True)
    mean_neighbors = torch.matmul(adjacency, x) / degree.clamp_min(1.0)

    if aggregator == "mean":
        return mean_neighbors

    masked_neighbors = x.unsqueeze(1).expand(-1, adjacency.size(1), -1, -1)
    mask = adjacency.unsqueeze(-1) > 0
    masked_neighbors = masked_neighbors.masked_fill(~mask, torch.finfo(x.dtype).min)
    max_neighbors = masked_neighbors.max(dim=2).values
    max_neighbors = torch.where(degree > 0, max_neighbors, torch.zeros_like(max_neighbors))

    if aggregator == "max":
        return max_neighbors
    if aggregator == "meanmax":
        return torch.cat([mean_neighbors, max_neighbors], dim=-1)
    raise ValueError(f"Unsupported GraphSAGE aggregator: {aggregator}")


class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2, aggregator: str = "meanmax") -> None:
        super().__init__()
        self.aggregator = aggregator
        neighbor_dim = in_dim if aggregator in {"mean", "max"} else 2 * in_dim
        self.self_proj = KANLinear(in_dim, out_dim)
        self.neighbor_proj = KANLinear(neighbor_dim, out_dim)
        self.output_proj = KANLinear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        neighborhood = _aggregate_neighbors(x, adjacency, self.aggregator)
        out = self.self_proj(x) + self.neighbor_proj(neighborhood)
        out = self.output_proj(out)
        out = F.gelu(out)
        return self.norm(self.dropout(out))


class HeteroGraphSAGEBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.2, aggregator: str = "meanmax") -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [GraphSAGELayer(hidden_dim, hidden_dim, dropout=dropout, aggregator=aggregator) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x, adjacency)
        return x


class HeteroGraphSAGELayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2, aggregator: str = "meanmax") -> None:
        super().__init__()
        self.aggregator = aggregator
        neighbor_dim = in_dim if aggregator in {"mean", "max"} else 2 * in_dim

        self.proj_type1 = KANLinear(in_dim, in_dim)
        self.proj_type2 = KANLinear(in_dim, in_dim)

        self.self_proj1 = KANLinear(in_dim, out_dim)
        self.self_proj2 = KANLinear(in_dim, out_dim)
        self.same_proj1 = KANLinear(neighbor_dim, out_dim)
        self.same_proj2 = KANLinear(neighbor_dim, out_dim)
        self.cross_proj12 = KANLinear(neighbor_dim, out_dim)
        self.cross_proj21 = KANLinear(neighbor_dim, out_dim)

        self.master_self_proj = KANLinear(in_dim, out_dim)
        self.master_node_proj = KANLinear(neighbor_dim, out_dim)
        self.master_gate = KANLinear(in_dim * 2, 1)

        self.norm_type1 = nn.LayerNorm(out_dim)
        self.norm_type2 = nn.LayerNorm(out_dim)
        self.norm_master = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def _update_master(self, x: torch.Tensor, master: torch.Tensor) -> torch.Tensor:
        if master.size(0) == 1 and x.size(0) > 1:
            master = master.expand(x.size(0), -1, -1)
        expanded_master = master.expand(-1, x.size(1), -1)
        gates = torch.sigmoid(self.master_gate(torch.cat([x, expanded_master], dim=-1)))
        weighted_nodes = x * gates
        mean_context = weighted_nodes.sum(dim=1, keepdim=True) / gates.sum(dim=1, keepdim=True).clamp_min(1e-6)
        if self.aggregator == "mean":
            context = mean_context
        else:
            max_context = weighted_nodes.max(dim=1, keepdim=True).values
            if self.aggregator == "max":
                context = max_context
            elif self.aggregator == "meanmax":
                context = torch.cat([mean_context, max_context], dim=-1)
            else:
                raise ValueError(f"Unsupported GraphSAGE aggregator: {self.aggregator}")
        master = self.master_self_proj(master) + self.master_node_proj(context)
        return self.norm_master(self.dropout(F.gelu(master)))

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        adjacency: torch.Tensor,
        master: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)

        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = x.mean(dim=1, keepdim=True)

        adj11 = adjacency[:, :num_type1, :num_type1]
        adj12 = adjacency[:, :num_type1, num_type1:]
        adj21 = adjacency[:, num_type1:, :num_type1]
        adj22 = adjacency[:, num_type1:, num_type1:]

        same_type1 = _aggregate_neighbors(x1, adj11, self.aggregator)
        cross_type1 = _aggregate_neighbors(x2, adj12, self.aggregator)
        same_type2 = _aggregate_neighbors(x2, adj22, self.aggregator)
        cross_type2 = _aggregate_neighbors(x1, adj21, self.aggregator)

        out1 = self.self_proj1(x1) + self.same_proj1(same_type1) + self.cross_proj12(cross_type1)
        out2 = self.self_proj2(x2) + self.same_proj2(same_type2) + self.cross_proj21(cross_type2)

        out1 = self.norm_type1(self.dropout(F.gelu(out1)))
        out2 = self.norm_type2(self.dropout(F.gelu(out2)))
        master = self._update_master(x, master)
        return out1, out2, master


class GraphSAGEInferenceBranch(nn.Module):
    def __init__(
        self,
        gat_dims,
        pool_ratio: float,
        graph_builder_factory,
        dropout: float = 0.2,
        aggregator: str = "meanmax",
    ) -> None:
        super().__init__()
        self.stage1 = HeteroGraphSAGELayer(gat_dims[0], gat_dims[1], dropout=dropout, aggregator=aggregator)
        self.stage2 = HeteroGraphSAGELayer(gat_dims[1], gat_dims[1], dropout=dropout, aggregator=aggregator)

        self.pool_hS = GraphPool(pool_ratio, gat_dims[1], 0.3, size=0)
        self.pool_hT = GraphPool(pool_ratio, gat_dims[1], 0.3, size=0)
        self.builder_stage1 = graph_builder_factory(gat_dims[0])
        self.builder_stage2 = graph_builder_factory(gat_dims[1])

    def _build_joint_graph(self, out_T: torch.Tensor, out_S: torch.Tensor, builder) -> torch.Tensor:
        return builder(torch.cat([out_T, out_S], dim=1))

    def forward(self, out_T: torch.Tensor, out_S: torch.Tensor, master: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        adjacency_stage1 = self._build_joint_graph(out_T, out_S, self.builder_stage1)
        out_T_res, out_S_res, master_res = self.stage1(out_T, out_S, adjacency_stage1, master=master)

        out_S_res = self.pool_hS(out_S_res)
        out_T_res = self.pool_hT(out_T_res)

        adjacency_stage2 = self._build_joint_graph(out_T_res, out_S_res, self.builder_stage2)
        out_T_aug, out_S_aug, master_aug = self.stage2(out_T_res, out_S_res, adjacency_stage2, master=master_res)

        out_T_final = out_T_res + out_T_aug
        out_S_final = out_S_res + out_S_aug
        master_final = master_res + master_aug
        return out_T_final, out_S_final, master_final
