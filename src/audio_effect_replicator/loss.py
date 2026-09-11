"""Training losses, all computed over the last `timesteps` samples of each window."""

from collections.abc import Callable

import torch
from torch.nn.functional import mse_loss

from audio_effect_replicator.config import LossSpec

PREEMPHASIS = 0.85  # the first-order high-pass of Wright et al., DAFx 2019

Loss = Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor]


def tail_mse(y_pred: torch.Tensor, y_true: torch.Tensor, timesteps: int) -> torch.Tensor:
    return mse_loss(y_pred[:, -timesteps:], y_true[:, -timesteps:])


def esr(y_pred: torch.Tensor, y_true: torch.Tensor, timesteps: int) -> torch.Tensor:
    """Error-to-signal ratio over the batch: squared error divided by the target's energy."""
    pred, true = y_pred[:, -timesteps:], y_true[:, -timesteps:]
    energy = torch.sum(true**2)
    if energy == 0:
        raise ValueError("target is silent, ESR is undefined")
    return torch.sum((pred - true) ** 2) / energy


def esr_preemphasis(y_pred: torch.Tensor, y_true: torch.Tensor, timesteps: int) -> torch.Tensor:
    """ESR of the high-passed signals plus a DC term, the DAFx 2019 training loss."""
    pred, true = y_pred[:, -timesteps:], y_true[:, -timesteps:]
    filtered_pred, filtered_true = _preemphasise(pred), _preemphasise(true)
    energy = torch.sum(filtered_true**2)
    if energy == 0:
        raise ValueError("target is silent, ESR is undefined")
    ratio = torch.sum((filtered_pred - filtered_true) ** 2) / energy
    dc = torch.mean(true - pred) ** 2 / torch.mean(true**2)
    return ratio + dc


def build_loss(spec: LossSpec) -> Loss:
    losses: dict[str, Loss] = {
        "tail_mse": tail_mse,
        "esr": esr,
        "esr_preemphasis": esr_preemphasis,
    }
    if spec.type not in losses:
        raise ValueError(f"unknown loss type {spec.type!r}; use {', '.join(losses)}")
    return losses[spec.type]


def _preemphasise(x: torch.Tensor) -> torch.Tensor:
    return x[:, 1:] - PREEMPHASIS * x[:, :-1]
