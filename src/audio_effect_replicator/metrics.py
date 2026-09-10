"""Quality and size metrics for comparing a predicted waveform with its target."""

import numpy as np
import torch
from torch import nn


def mse(pred: np.ndarray, target: np.ndarray) -> float:
    _check(pred, target)
    return float(np.mean((pred - target) ** 2))


def esr(pred: np.ndarray, target: np.ndarray) -> float:
    """Error-to-signal ratio: sum of squared error over the target's energy."""
    _check(pred, target)
    energy = float(np.sum(target**2))
    if energy == 0.0:
        raise ValueError("target is silent, ESR is undefined")
    return float(np.sum((pred - target) ** 2) / energy)


def log_spectral_distance(
    pred: np.ndarray, target: np.ndarray, n_fft: int = 2048, hop: int = 512
) -> float:
    """RMS difference of the log-magnitude STFTs, in dB."""
    _check(pred, target)
    window = torch.hann_window(n_fft)

    def log_magnitude(x: np.ndarray) -> torch.Tensor:
        spec = torch.stft(
            torch.from_numpy(np.asarray(x, np.float32)),
            n_fft,
            hop,
            window=window,
            return_complex=True,
        )
        return 20 * torch.log10(spec.abs() + 1e-8)

    return float(torch.sqrt(torch.mean((log_magnitude(pred) - log_magnitude(target)) ** 2)))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _check(pred: np.ndarray, target: np.ndarray) -> None:
    if len(target) == 0:
        raise ValueError("signals must not be empty")
    if len(pred) != len(target):
        raise ValueError(f"length mismatch: {len(pred)} vs {len(target)}")
