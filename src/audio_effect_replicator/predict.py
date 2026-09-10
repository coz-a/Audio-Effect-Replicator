"""Sliding-window inference (window = input_timesteps, hop = output_timesteps).

Unlike the 2018 script, the tail padding is rounded up so the output always has
exactly the input length.
"""

import numpy as np
import torch
from torch import nn


def predict(
    model: nn.Module,
    samples: np.ndarray,
    input_timesteps: int,
    output_timesteps: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    n = len(samples)
    if n == 0:
        return np.zeros(0, np.float32)

    prepad = input_timesteps - output_timesteps
    postpad = (-n) % output_timesteps
    padded = np.concatenate(
        [np.zeros(prepad, np.float32), samples.astype(np.float32), np.zeros(postpad, np.float32)]
    )
    windows = torch.from_numpy(padded).unfold(0, input_timesteps, output_timesteps).unsqueeze(-1)

    model.eval()
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch = windows[start : start + batch_size].to(device)
            outputs.append(model(batch)[:, -output_timesteps:, 0].cpu())
    return torch.cat(outputs).reshape(-1).numpy()[:n]
