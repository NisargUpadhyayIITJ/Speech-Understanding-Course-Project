from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graphs import DynamicGraphBuilder, GraphSAGELayer, GraphSAGEInferenceBranch, MaskedGraphAttentionLayer
from ..kan import KANLinear
from ..pool import GraphPool
from ..residual import Residual_block


class LearnableGraphHead(nn.Module):
    """Graph experiment head that supports fixed/learned topology and GAT/GraphSAGE."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        num_nodes: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        graph_config: dict | None = None,
        size: int = 200,
        d_args: dict | None = None,
    ) -> None:
        super().__init__()
        self.graph_config = graph_config or {}
        self.operator = self.graph_config.get("operator", "gat")
        if self.operator == "graphsage":
            self.operator = "graphsage"
        self.dropout = nn.Dropout(dropout)
        self.last_graph_stats: dict[str, float] = {}

        if self.operator == "graphsage":
            self._init_graphsage_path(input_dim, hidden_dim, num_classes, dropout, size, d_args)
        else:
            self._init_dense_gat_path(input_dim, hidden_dim, num_classes, num_nodes, num_layers, dropout)

    def _build_graph_builder(self, hidden_dim: int) -> DynamicGraphBuilder:
        return DynamicGraphBuilder(
            hidden_dim=hidden_dim,
            mode=self.graph_config.get("mode", "fixed"),
            edge_score=self.graph_config.get("edge_score", "dot"),
            topk=self.graph_config.get("topk", 8),
            symmetric=self.graph_config.get("symmetric", True),
            include_self_loops=True,
        )

    def _init_dense_gat_path(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_nodes: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        self.bridge = KANLinear(input_dim, hidden_dim)
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SELU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SELU(inplace=True),
        )
        self.node_pool = nn.AdaptiveAvgPool1d(num_nodes)
        self.positional_embedding = nn.Parameter(torch.randn(1, num_nodes, hidden_dim))
        self.graph_builder = self._build_graph_builder(hidden_dim)
        self.layers = nn.ModuleList([MaskedGraphAttentionLayer(hidden_dim, hidden_dim, dropout=dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim * 2
        self.classifier = KANLinear(self.output_dim, num_classes)

    def _init_graphsage_path(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        size: int,
        d_args: dict | None,
    ) -> None:
        self.d_args = d_args or {
            "architecture": "AASIST",
            "nb_samp": 64600,
            "first_conv": hidden_dim,
            "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
            "gat_dims": [64, 32],
            "pool_ratios": [0.5, 0.7, 0.5, 0.5],
            "temperatures": [2.0, 2.0, 100.0, 100.0],
        }

        filts = self.d_args["filts"]
        gat_dims = self.d_args["gat_dims"]
        pool_ratios = self.d_args["pool_ratios"]
        aggregator = self.graph_config.get("aggregator", "meanmax")

        self.bridge = KANLinear(input_dim, hidden_dim)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )

        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        self.pos_T = nn.Parameter(torch.randn(1, 67, filts[-1][-1]))

        self.graphsage_layer_S = GraphSAGELayer(filts[-1][-1], gat_dims[0], dropout=dropout, aggregator=aggregator)
        self.graphsage_layer_T = GraphSAGELayer(filts[-1][-1], gat_dims[0], dropout=dropout, aggregator=aggregator)
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3, size=size)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3, size=size)
        self.builder_S = self._build_graph_builder(filts[-1][-1])
        self.builder_T = self._build_graph_builder(filts[-1][-1])

        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master3 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master4 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        branch_factory = lambda _: self._build_graph_builder(_)
        self.inference_branch1 = GraphSAGEInferenceBranch(gat_dims=gat_dims, pool_ratio=pool_ratios[2], graph_builder_factory=branch_factory, dropout=dropout, aggregator=aggregator)
        self.inference_branch2 = GraphSAGEInferenceBranch(gat_dims=gat_dims, pool_ratio=pool_ratios[2], graph_builder_factory=branch_factory, dropout=dropout, aggregator=aggregator)
        self.inference_branch3 = GraphSAGEInferenceBranch(gat_dims=gat_dims, pool_ratio=pool_ratios[2], graph_builder_factory=branch_factory, dropout=dropout, aggregator=aggregator)
        self.inference_branch4 = GraphSAGEInferenceBranch(gat_dims=gat_dims, pool_ratio=pool_ratios[2], graph_builder_factory=branch_factory, dropout=dropout, aggregator=aggregator)

        self.output_dim = 5 * gat_dims[1]
        self.classifier = KANLinear(self.output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.operator == "graphsage":
            return self._forward_graphsage(x)
        return self._forward_dense_gat(x)

    def _collect_builder_stats(self, prefix: str, builder: DynamicGraphBuilder) -> dict[str, float]:
        if not getattr(builder, "last_stats", None):
            return {}
        return {f"{prefix}_{key}": value for key, value in builder.last_stats.items()}

    def _builder_raw_stats(self, builder: DynamicGraphBuilder) -> dict[str, float]:
        return dict(getattr(builder, "last_stats", {}) or {})

    def _aggregate_stat_groups(self, stats_list: list[dict[str, float]]) -> dict[str, float]:
        aggregate: dict[str, list[float]] = {}
        for stats in stats_list:
            for key, value in stats.items():
                aggregate.setdefault(key, []).append(value)
        return {key: sum(values) / len(values) for key, values in aggregate.items() if values}

    def _forward_dense_gat(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.bridge(x)
        x = self.temporal_encoder(x.transpose(1, 2))
        x = self.node_pool(x).transpose(1, 2)
        x = x + self.positional_embedding
        adjacency = self.graph_builder(x)
        self.last_graph_stats = self._collect_builder_stats("graph", self.graph_builder)
        for layer in self.layers:
            x = x + layer(x, adjacency)
        x = self.norm(x)
        pooled_mean = x.mean(dim=1)
        pooled_max = x.max(dim=1).values
        embedding = torch.cat([pooled_mean, pooled_max], dim=1)
        logits = self.classifier(self.dropout(embedding))
        return logits, embedding

    def _forward_graphsage(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.bridge(x)
        x = x.transpose(1, 2)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        e = self.encoder(x)

        e_S, _ = torch.max(torch.abs(e), dim=3)
        e_S = e_S.transpose(1, 2) + self.pos_S
        adj_S = self.builder_S(e_S)
        out_S = self.pool_S(self.graphsage_layer_S(e_S, adj_S))

        e_T, _ = torch.max(torch.abs(e), dim=2)
        e_T = e_T.transpose(1, 2) + self.pos_T
        adj_T = self.builder_T(e_T)
        out_T = self.pool_T(self.graphsage_layer_T(e_T, adj_T))

        out_T1, out_S1, master1 = self.inference_branch1(out_T, out_S, self.master1)
        out_T2, out_S2, master2 = self.inference_branch2(out_T, out_S, self.master2)
        out_T3, out_S3, master3 = self.inference_branch3(out_T, out_S, self.master3)
        out_T4, out_S4, master4 = self.inference_branch4(out_T, out_S, self.master4)

        out_T1, out_T2 = self.drop_way(out_T1), self.drop_way(out_T2)
        out_T3, out_T4 = self.drop_way(out_T3), self.drop_way(out_T4)
        out_S1, out_S2 = self.drop_way(out_S1), self.drop_way(out_S2)
        out_S3, out_S4 = self.drop_way(out_S3), self.drop_way(out_S4)
        master1, master2 = self.drop_way(master1), self.drop_way(master2)
        master3, master4 = self.drop_way(master3), self.drop_way(master4)

        out_T = torch.stack([out_T1, out_T2, out_T3, out_T4]).max(dim=0)[0]
        out_S = torch.stack([out_S1, out_S2, out_S3, out_S4]).max(dim=0)[0]
        master = torch.stack([master1, master2, master3, master4]).max(dim=0)[0]

        T_max = torch.max(torch.abs(out_T), dim=1).values
        T_avg = out_T.mean(dim=1)
        S_max = torch.max(torch.abs(out_S), dim=1).values
        S_avg = out_S.mean(dim=1)

        embedding = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        branch_stats = [
            self._collect_builder_stats("spatial", self.builder_S),
            self._collect_builder_stats("temporal", self.builder_T),
            self._collect_builder_stats("branch1_stage1", self.inference_branch1.builder_stage1),
            self._collect_builder_stats("branch1_stage2", self.inference_branch1.builder_stage2),
            self._collect_builder_stats("branch2_stage1", self.inference_branch2.builder_stage1),
            self._collect_builder_stats("branch2_stage2", self.inference_branch2.builder_stage2),
            self._collect_builder_stats("branch3_stage1", self.inference_branch3.builder_stage1),
            self._collect_builder_stats("branch3_stage2", self.inference_branch3.builder_stage2),
            self._collect_builder_stats("branch4_stage1", self.inference_branch4.builder_stage1),
            self._collect_builder_stats("branch4_stage2", self.inference_branch4.builder_stage2),
        ]
        summary_sources = [
            self._builder_raw_stats(self.builder_S),
            self._builder_raw_stats(self.builder_T),
            self._builder_raw_stats(self.inference_branch1.builder_stage1),
            self._builder_raw_stats(self.inference_branch1.builder_stage2),
            self._builder_raw_stats(self.inference_branch2.builder_stage1),
            self._builder_raw_stats(self.inference_branch2.builder_stage2),
            self._builder_raw_stats(self.inference_branch3.builder_stage1),
            self._builder_raw_stats(self.inference_branch3.builder_stage2),
            self._builder_raw_stats(self.inference_branch4.builder_stage1),
            self._builder_raw_stats(self.inference_branch4.builder_stage2),
        ]
        summary_stats = self._aggregate_stat_groups([stats for stats in summary_sources if stats])
        prefixed_summary = {f"graph_summary_{key}": value for key, value in summary_stats.items()}
        self.last_graph_stats = {**prefixed_summary}
        for stats in branch_stats:
            self.last_graph_stats.update(stats)
        logits = self.classifier(self.dropout(embedding))
        return logits, embedding
