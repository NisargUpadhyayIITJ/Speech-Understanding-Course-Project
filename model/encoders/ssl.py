from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class SSLEncoder(nn.Module):
    """Configurable SSL encoder wrapper used across all experiment families."""

    def __init__(
        self,
        name: str = "wav2vec2",
        model_name: str = "facebook/wav2vec2-large-xlsr-53",
        load_pretrained: bool = True,
        normalize_waveform: bool = True,
        cache_dir: str = "weights",
        freeze_strategy: str = "none",
        unfreeze_last_n_layers: int = 0,
        layer_aggregation: str = "last",
        mean_last_k: int = 4,
    ) -> None:
        super().__init__()
        self.name = name
        self.model_name = model_name
        self.normalize_waveform = normalize_waveform
        self.freeze_strategy = freeze_strategy
        self.unfreeze_last_n_layers = unfreeze_last_n_layers
        self.layer_aggregation = layer_aggregation
        self.mean_last_k = mean_last_k

        if load_pretrained:
            self.model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        else:
            try:
                config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=False)
            except Exception:
                config = AutoConfig.for_model(name)
            self.model = AutoModel.from_config(config)

        self.output_dim = self.model.config.hidden_size
        self._apply_freeze_strategy()

    def _apply_freeze_strategy(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = True

        if self.freeze_strategy == "none":
            return

        if self.freeze_strategy == "full":
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            return

        if self.freeze_strategy != "partial":
            raise ValueError(f"Unknown freeze strategy: {self.freeze_strategy}")

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        encoder_layers = getattr(self.model.encoder, "layers", None)
        if encoder_layers is None:
            return

        num_layers = len(encoder_layers)
        start_idx = max(num_layers - self.unfreeze_last_n_layers, 0)
        for layer in encoder_layers[start_idx:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

        for module_name in ("feature_projection", "encoder.layer_norm", "masked_spec_embed"):
            module = self._resolve_attr_path(module_name)
            if module is None:
                continue
            for parameter in module.parameters():
                parameter.requires_grad = True

    def _resolve_attr_path(self, attr_path: str) -> Optional[nn.Module]:
        current = self.model
        for attr_name in attr_path.split("."):
            if not hasattr(current, attr_name):
                return None
            current = getattr(current, attr_name)
        return current

    def _select_hidden_states(self, outputs) -> torch.Tensor:
        if self.layer_aggregation == "last":
            return outputs.last_hidden_state

        if self.layer_aggregation != "mean_last_k":
            raise ValueError(f"Unknown layer aggregation mode: {self.layer_aggregation}")

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("mean_last_k aggregation requires hidden states from the encoder")

        selected = hidden_states[-self.mean_last_k:]
        return torch.mean(torch.stack(selected, dim=0), dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        elif x.ndim == 3 and x.size(-1) == 1:
            x = x.squeeze(-1)

        if self.normalize_waveform:
            x = x / (torch.max(torch.abs(x), dim=1, keepdim=True)[0] + 1e-8)

        outputs = self.model(
            x,
            output_hidden_states=self.layer_aggregation == "mean_last_k",
            return_dict=True,
        )
        features = self._select_hidden_states(outputs)
        del outputs
        return features
