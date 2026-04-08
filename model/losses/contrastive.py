from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("SupervisedContrastiveLoss expects features of shape [batch, views, dim]")

        batch_size, num_views, hidden_dim = features.shape
        features = F.normalize(features.reshape(batch_size * num_views, hidden_dim), dim=-1)
        expanded_labels = labels.repeat_interleave(num_views)
        similarity = torch.matmul(features, features.transpose(0, 1)) / self.temperature

        logits_mask = ~torch.eye(similarity.size(0), dtype=torch.bool, device=similarity.device)
        positive_mask = expanded_labels.unsqueeze(0) == expanded_labels.unsqueeze(1)
        positive_mask = positive_mask & logits_mask

        similarity = similarity.masked_fill(~logits_mask, torch.finfo(similarity.dtype).min)
        log_prob = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)

        positives_per_sample = positive_mask.sum(dim=1).clamp_min(1)
        mean_log_prob = (positive_mask * log_prob).sum(dim=1) / positives_per_sample
        valid = positive_mask.sum(dim=1) > 0
        return -mean_log_prob[valid].mean()


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)
        logits = torch.matmul(query, key.transpose(0, 1)) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        forward_loss = F.cross_entropy(logits, labels)
        backward_loss = F.cross_entropy(logits.transpose(0, 1), labels)
        return 0.5 * (forward_loss + backward_loss)


class TrainingObjective(nn.Module):
    def __init__(
        self,
        name: str = "ce",
        ce_weight: float = 1.0,
        contrastive_weight: float = 0.0,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.name = name
        self.ce_weight = ce_weight
        self.contrastive_weight = contrastive_weight
        self.cross_entropy = nn.CrossEntropyLoss()
        self.supcon = SupervisedContrastiveLoss(temperature=temperature)
        self.infonce = InfoNCELoss(temperature=temperature)

    @property
    def requires_embedding_outputs(self) -> bool:
        return self.contrastive_weight > 0

    @property
    def requires_two_views(self) -> bool:
        return self.name in {"ce_supcon", "ce_infonce"}

    def _extract(self, output, key: str):
        if not isinstance(output, dict):
            if key == "logits":
                return output
            return None
        return output.get(key)

    def forward(
        self,
        primary_output,
        targets: torch.Tensor,
        secondary_output=None,
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        logits = self._extract(primary_output, "logits")
        ce_loss = self.cross_entropy(logits, targets)

        if secondary_output is not None:
            ce_loss = 0.5 * (ce_loss + self.cross_entropy(self._extract(secondary_output, "logits"), targets))

        total_loss = self.ce_weight * ce_loss
        metrics: Dict[str, torch.Tensor] = {"ce_loss": ce_loss.detach()}

        if self.contrastive_weight <= 0:
            metrics["total_loss"] = total_loss.detach()
            return total_loss, metrics

        projection = self._extract(primary_output, "projection")
        second_projection = self._extract(secondary_output, "projection") if secondary_output is not None else None

        if projection is None:
            raise RuntimeError("Contrastive objectives require model outputs to include `projection`.")
        if self.requires_two_views and second_projection is None:
            raise RuntimeError(f"{self.name} requires batches with two views per utterance.")

        if self.name == "ce_supcon":
            features = torch.stack([projection, second_projection], dim=1)
            contrastive_loss = self.supcon(features, targets)
        elif self.name == "ce_infonce":
            contrastive_loss = self.infonce(projection, second_projection)
        else:
            raise ValueError(f"Unknown training objective: {self.name}")

        total_loss = total_loss + self.contrastive_weight * contrastive_loss
        metrics["contrastive_loss"] = contrastive_loss.detach()
        metrics["total_loss"] = total_loss.detach()
        return total_loss, metrics


def build_training_objective(config: Optional[dict]) -> TrainingObjective:
    config = config or {}
    return TrainingObjective(
        name=config.get("name", "ce"),
        ce_weight=config.get("ce_weight", 1.0),
        contrastive_weight=config.get("contrastive_weight", 0.0),
        temperature=config.get("temperature", 0.07),
    )

