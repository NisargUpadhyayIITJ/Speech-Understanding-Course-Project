from __future__ import annotations

from copy import deepcopy

from .encoders import SSLEncoder
from .heads import AASISTGraphHead, KANAttentionHead, LearnableGraphHead, MambaHead
from .losses import ProjectionHead


DEFAULT_ENCODER_CONFIG = {
    "name": "wav2vec2",
    "model_name": "facebook/wav2vec2-large-xlsr-53",
    "load_pretrained": True,
    "normalize_waveform": True,
    "cache_dir": "weights",
    "freeze_strategy": "none",
    "unfreeze_last_n_layers": 0,
    "layer_aggregation": "last",
    "mean_last_k": 4,
}

DEFAULT_MODEL_CONFIG = {
    "name": "baseline",
    "hidden_dim": 128,
    "num_classes": 2,
    "dropout": 0.5,
    "branch_dropout": 0.2,
    "size": 200,
    "num_layers": 2,
    "num_heads": 4,
    "mlp_ratio": 4,
    "use_kan_feedforward": True,
    "num_nodes": 64,
    "state_dim": 16,
    "conv_kernel": 4,
    "expand": 2,
}

DEFAULT_GRAPH_CONFIG = {
    "mode": "fixed",
    "operator": "gat",
    "edge_score": "dot",
    "topk": 8,
    "symmetric": True,
    "dropout": 0.2,
    "aggregator": "meanmax",
}

DEFAULT_PROJECTION_CONFIG = {
    "enabled": False,
    "hidden_dim": 256,
    "dim": 128,
    "dropout": 0.1,
}


def deep_merge(base: dict, overrides: dict | None) -> dict:
    merged = deepcopy(base)
    if not overrides:
        return merged

    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_encoder(config: dict | None, legacy_cache_dir: str = "weights", legacy_load_pretrained: bool = True) -> SSLEncoder:
    encoder_config = deep_merge(DEFAULT_ENCODER_CONFIG, config)
    encoder_config["cache_dir"] = encoder_config.get("cache_dir", legacy_cache_dir)
    encoder_config["load_pretrained"] = encoder_config.get("load_pretrained", legacy_load_pretrained)

    if encoder_config.get("name") not in {"wav2vec2", "hubert"}:
        raise ValueError(f"Unsupported encoder: {encoder_config.get('name')}")

    return SSLEncoder(
        name=encoder_config["name"],
        model_name=encoder_config["model_name"],
        load_pretrained=encoder_config["load_pretrained"],
        normalize_waveform=encoder_config["normalize_waveform"],
        cache_dir=encoder_config["cache_dir"],
        freeze_strategy=encoder_config["freeze_strategy"],
        unfreeze_last_n_layers=encoder_config["unfreeze_last_n_layers"],
        layer_aggregation=encoder_config["layer_aggregation"],
        mean_last_k=encoder_config["mean_last_k"],
    )


def build_head(
    config: dict | None,
    input_dim: int,
    graph_config: dict | None = None,
    legacy_d_args: dict | None = None,
    legacy_size: int = 200,
):
    model_config = deep_merge(DEFAULT_MODEL_CONFIG, config)
    graph_config = deep_merge(DEFAULT_GRAPH_CONFIG, graph_config)
    name = model_config.get("name", "baseline")

    if name in {"baseline", "aasist_graph"}:
        return AASISTGraphHead(
            input_dim=input_dim,
            hidden_dim=model_config["hidden_dim"],
            num_classes=model_config["num_classes"],
            size=model_config.get("size", legacy_size),
            d_args=legacy_d_args,
            dropout=model_config["dropout"],
            branch_dropout=model_config["branch_dropout"],
        )

    if name == "learnable_graph":
        return LearnableGraphHead(
            input_dim=input_dim,
            hidden_dim=model_config["hidden_dim"],
            num_classes=model_config["num_classes"],
            num_nodes=model_config["num_nodes"],
            num_layers=model_config["num_layers"],
            dropout=graph_config["dropout"],
            graph_config=graph_config,
            size=model_config.get("size", legacy_size),
            d_args=legacy_d_args,
        )

    if name == "kan_attention":
        return KANAttentionHead(
            input_dim=input_dim,
            hidden_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            num_heads=model_config["num_heads"],
            mlp_ratio=model_config["mlp_ratio"],
            dropout=model_config["dropout"],
            num_classes=model_config["num_classes"],
            use_kan_feedforward=model_config["use_kan_feedforward"],
        )

    if name == "mamba":
        return MambaHead(
            input_dim=input_dim,
            hidden_dim=model_config["hidden_dim"],
            num_layers=model_config["num_layers"],
            state_dim=model_config["state_dim"],
            conv_kernel=model_config["conv_kernel"],
            expand=model_config["expand"],
            dropout=model_config["dropout"],
            num_classes=model_config["num_classes"],
        )

    raise ValueError(f"Unsupported model head: {name}")


def build_projection_head(config: dict | None, input_dim: int):
    projection_config = deep_merge(DEFAULT_PROJECTION_CONFIG, config)
    if not projection_config.get("enabled", False):
        return None
    return ProjectionHead(
        input_dim=input_dim,
        hidden_dim=projection_config["hidden_dim"],
        output_dim=projection_config["dim"],
        dropout=projection_config["dropout"],
    )
