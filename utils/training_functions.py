import torch

from tqdm import tqdm


def _extract_graph_diagnostics(model, accelerator):
    unwrapped_model = accelerator.unwrap_model(model)
    head = getattr(unwrapped_model, "head", None)
    diagnostics = getattr(head, "last_graph_stats", None)
    if not diagnostics:
        return {}

    averaged = {}
    for key, value in diagnostics.items():
        tensor = accelerator.gather_for_metrics(torch.tensor([float(value)], device=accelerator.device))
        averaged[key] = tensor.float().mean().item()
    return averaged


def _average_diagnostic_groups(*groups):
    aggregate = {}
    for group in groups:
        if not group:
            continue
        for key, value in group.items():
            aggregate.setdefault(key, []).append(value)
    return {key: sum(values) / len(values) for key, values in aggregate.items() if values}


def _accumulate_metric_sums(metric_sums, metrics):
    if not metrics:
        return
    for key, value in metrics.items():
        metric_sums[key] = metric_sums.get(key, 0.0) + float(value)


def train_one_epoch(model, dataloader, criterion, optimizer, accelerator, max_batches=-1):
    """
    Trains the model for one epoch.
    Args:
        model: The neural network to train.
        dataloader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        accelerator: Accelerator (from HuggingFace Accelerate).
        max_batches: Maximum number of batches to process.
    Returns:
        Average loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    epoch_metric_sums = {}
    for batch in tqdm(dataloader, desc="Training", leave=False):
        if len(batch) == 2:
            inputs, targets = batch
            second_inputs = None
        elif len(batch) >= 3:
            inputs, second_inputs, targets = batch[:3]
        else:
            raise ValueError("Unexpected batch structure received from the dataloader")

        with accelerator.accumulate(model):
            if second_inputs is not None:
                combined_inputs = torch.cat([inputs, second_inputs], dim=0)
                combined_outputs = model(combined_inputs, return_embedding=criterion.requires_embedding_outputs)
                primary_graph_metrics = _extract_graph_diagnostics(model, accelerator)
                secondary_graph_metrics = {}
                batch_size = inputs.size(0)
                if isinstance(combined_outputs, dict):
                    outputs = {k: v[:batch_size] for k, v in combined_outputs.items()}
                    second_outputs = {k: v[batch_size:] for k, v in combined_outputs.items()}
                else:
                    outputs = combined_outputs[:batch_size]
                    second_outputs = combined_outputs[batch_size:]
            else:
                outputs = model(inputs, return_embedding=criterion.requires_embedding_outputs)
                primary_graph_metrics = _extract_graph_diagnostics(model, accelerator)
                second_outputs = None
                secondary_graph_metrics = {}
            loss, loss_metrics = criterion(outputs, targets, second_outputs)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item()
        graph_metrics = _average_diagnostic_groups(primary_graph_metrics, secondary_graph_metrics)
        batch_metrics = {
            "loss": loss.item(),
            **{key: value.item() for key, value in loss_metrics.items()},
            **graph_metrics,
        }
        _accumulate_metric_sums(epoch_metric_sums, batch_metrics)
        accelerator.log(
            batch_metrics
        )
        num_batches += 1
        if (max_batches != -1) and (num_batches > max_batches):
            break

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_metrics = {
        f"epoch_avg/{key}": value / num_batches
        for key, value in epoch_metric_sums.items()
    } if num_batches > 0 else {}
    return avg_loss, avg_metrics
