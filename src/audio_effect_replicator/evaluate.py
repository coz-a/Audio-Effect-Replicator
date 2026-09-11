"""Score a prediction against its target, and time the sliding-window inference."""

import time

import numpy as np
import torch
from torch import nn

from audio_effect_replicator.audio import SAMPLE_RATE
from audio_effect_replicator.metrics import esr, log_spectral_distance, mse
from audio_effect_replicator.predict import predict


def evaluate_prediction(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    return {
        "mse": mse(pred, target),
        "esr": esr(pred, target),
        "lsd_db": log_spectral_distance(pred, target),
    }


def time_inference(
    model: nn.Module,
    samples: np.ndarray,
    input_timesteps: int,
    output_timesteps: int,
    batch_size: int,
    device: torch.device,
    sample_rate: int = SAMPLE_RATE,
) -> dict[str, float]:
    """Wall-clock time of one `predict` call after one warm-up call."""
    predict(model, samples, input_timesteps, output_timesteps, batch_size, device)
    _synchronize(device)
    start = time.perf_counter()
    predict(model, samples, input_timesteps, output_timesteps, batch_size, device)
    _synchronize(device)
    seconds = time.perf_counter() - start
    return {"inference_seconds": seconds, "realtime_factor": len(samples) / sample_rate / seconds}


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
