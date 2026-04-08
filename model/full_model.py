#!/usr/bin/env python

from __future__ import annotations

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .factory import build_encoder, build_head, build_projection_head


class aasist3(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        model_config: dict | None = None,
        encoder_config: dict | None = None,
        graph_config: dict | None = None,
        projection_config: dict | None = None,
        return_embedding: bool = False,
        d_args: dict | None = None,
        size: int = 200,
        w2v_cache_dir: str = "weights/",
        load_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.return_embedding = return_embedding
        self.encoder = build_encoder(encoder_config, legacy_cache_dir=w2v_cache_dir, legacy_load_pretrained=load_pretrained)
        self.head = build_head(
            model_config,
            input_dim=self.encoder.output_dim,
            graph_config=graph_config,
            legacy_d_args=d_args,
            legacy_size=size,
        )
        self.projection_head = build_projection_head(projection_config, self.head.output_dim)

    def forward(self, x: torch.Tensor, return_embedding: bool | None = None):
        should_return_embeddings = self.return_embedding if return_embedding is None else return_embedding
        features = self.encoder(x)
        logits, embedding = self.head(features)

        if not should_return_embeddings and self.projection_head is None:
            return logits

        output = {"logits": logits, "embedding": embedding}
        if self.projection_head is not None:
            output["projection"] = self.projection_head(embedding)
        return output
