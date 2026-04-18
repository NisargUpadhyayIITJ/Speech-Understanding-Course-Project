from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Iterable

import soundfile as sf
import torch
import torchaudio
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from safetensors.torch import load_file


TARGET_SAMPLE_RATE = 16000
TARGET_NUM_SAMPLES = 64600
DEFAULT_CHECKPOINT_ROOT = Path("weights/train_fsage_mamba_supcon")
SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run anti-spoofing inference with the fsage+mamba+supcon checkpoint.",
    )
    parser.add_argument(
        "input_path",
        help="Path to an audio file or a directory of audio files.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a model.safetensors file. Defaults to the latest checkpoint under weights/train_fsage_mamba_supcon.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default=str(DEFAULT_CHECKPOINT_ROOT),
        help="Directory containing check_multigpu_epoch_* checkpoint folders.",
    )
    parser.add_argument(
        "--config-name",
        default="train",
        help="Hydra config name inside configs/. Default: train",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use, e.g. cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference. Larger values improve throughput on GPUs.",
    )
    parser.add_argument(
        "--num-segments",
        type=int,
        default=1,
        help="Number of evenly spaced fixed-length segments to score per file and average.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directories for audio files.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV path for saving predictions.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Force Hugging Face offline mode. Use this if the encoder is already cached locally.",
    )
    return parser.parse_args()


def set_offline_mode() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_project_config(config_name: str) -> dict:
    config_dir = Path(__file__).resolve().parent / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        config = compose(config_name=config_name)
    config_dict = OmegaConf.to_container(config, resolve=True)

    loss_config = config_dict.get("loss", {}) or {}
    projection_config = config_dict.get("projection", {}) or {}
    if loss_config.get("contrastive_weight", 0.0) > 0:
        projection_config["enabled"] = True
    config_dict["projection"] = projection_config
    return config_dict


def resolve_checkpoint(checkpoint: str | None, checkpoint_root: str) -> Path:
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        return checkpoint_path

    root = Path(checkpoint_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {root}")

    candidates = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"check_multigpu_epoch_(\d+)", child.name)
        if not match:
            continue
        model_file = child / "model.safetensors"
        if model_file.is_file():
            candidates.append((int(match.group(1)), model_file))

    if not candidates:
        raise FileNotFoundError(f"No model.safetensors files found under {root}")

    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def build_model(config: dict, checkpoint_path: Path, device: torch.device):
    from model import aasist3

    model = aasist3(
        model_config=config.get("model"),
        encoder_config=config.get("encoder"),
        graph_config=config.get("graph"),
        projection_config=config.get("projection"),
    )
    state_dict = load_file(str(checkpoint_path))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def collect_audio_files(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    iterator: Iterable[Path]
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    files = sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No supported audio files found in {input_path}. "
            f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return files


def load_audio(path: Path) -> torch.Tensor:
    audio_data, sample_rate = sf.read(str(path))
    audio = torch.from_numpy(audio_data).float()

    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.T

    if audio.size(0) > 1:
        audio = audio.mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sample_rate, TARGET_SAMPLE_RATE)
    audio = torchaudio.functional.preemphasis(audio)
    return audio.squeeze(0)


def build_segments(audio: torch.Tensor, target_length: int, num_segments: int) -> torch.Tensor:
    if audio.numel() == 0:
        raise ValueError("Encountered empty audio input.")
    if num_segments < 1:
        raise ValueError("--num-segments must be at least 1.")

    if audio.size(0) < target_length:
        repeat_factor = (target_length // audio.size(0)) + 1
        audio = audio.repeat(repeat_factor)

    if audio.size(0) == target_length:
        return audio.unsqueeze(0).repeat(num_segments, 1)

    max_start = audio.size(0) - target_length
    if num_segments == 1:
        starts = [max_start // 2]
    else:
        starts = torch.linspace(0, max_start, steps=num_segments).round().long().tolist()

    segments = [audio[start:start + target_length] for start in starts]
    return torch.stack(segments, dim=0)


@torch.inference_mode()
def score_batch(model, batch_waveforms: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(batch_waveforms)
    if isinstance(outputs, dict):
        outputs = outputs["logits"]
    probabilities = torch.softmax(outputs, dim=-1)
    return outputs, probabilities


def write_csv(output_path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "path",
        "predicted_label",
        "spoof_probability",
        "bonafide_probability",
        "spoof_logit",
        "bonafide_logit",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.offline:
        set_offline_mode()

    config = load_project_config(args.config_name)
    device = torch.device(args.device)
    checkpoint_path = resolve_checkpoint(args.checkpoint, args.checkpoint_root)
    model = build_model(config, checkpoint_path, device)

    audio_paths = collect_audio_files(Path(args.input_path), recursive=args.recursive)
    results = []

    for batch_start in range(0, len(audio_paths), args.batch_size):
        batch_paths = audio_paths[batch_start:batch_start + args.batch_size]
        batch_segments = []
        segment_counts = []

        for path in batch_paths:
            waveform = load_audio(path)
            segments = build_segments(
                waveform,
                target_length=TARGET_NUM_SAMPLES,
                num_segments=args.num_segments,
            )
            batch_segments.append(segments)
            segment_counts.append(segments.size(0))

        flat_batch = torch.cat(batch_segments, dim=0).to(device)
        logits, probabilities = score_batch(model, flat_batch)

        cursor = 0
        for path, segment_count in zip(batch_paths, segment_counts):
            path_logits = logits[cursor:cursor + segment_count].mean(dim=0).cpu()
            path_probs = probabilities[cursor:cursor + segment_count].mean(dim=0).cpu()
            cursor += segment_count

            spoof_probability = float(path_probs[0].item())
            bonafide_probability = float(path_probs[1].item())
            predicted_label = "bonafide" if bonafide_probability >= spoof_probability else "spoof"

            result = {
                "path": str(path),
                "predicted_label": predicted_label,
                "spoof_probability": f"{spoof_probability:.6f}",
                "bonafide_probability": f"{bonafide_probability:.6f}",
                "spoof_logit": f"{float(path_logits[0].item()):.6f}",
                "bonafide_logit": f"{float(path_logits[1].item()):.6f}",
            }
            results.append(result)
            print(
                f"{path} | predicted={predicted_label} | "
                f"spoof_prob={spoof_probability:.6f} | bonafide_prob={bonafide_probability:.6f}"
            )

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(output_path, results)
        print(f"\nSaved predictions to {output_path}")

    print(f"\nLoaded checkpoint: {checkpoint_path}")
    print(f"Processed {len(results)} file(s) on device {device}.")


if __name__ == "__main__":
    main()
