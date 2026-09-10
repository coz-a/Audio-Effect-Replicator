"""Random window batches from paired (dry, effected) waveforms, as in the 2018 generator."""

from pathlib import Path

import numpy as np
import torch

from audio_effect_replicator.audio import load_wave

Pair = tuple[np.ndarray, np.ndarray]


def load_pairs(paths: list[tuple[Path, Path]]) -> list[Pair]:
    pairs: list[Pair] = []
    for x_path, y_path in paths:
        x, y = load_wave(x_path), load_wave(y_path)
        if len(x) != len(y):
            raise ValueError(f"{x_path} and {y_path} differ in length ({len(x)} vs {len(y)})")
        pairs.append((x, y))
    return pairs


class WindowSampler:
    """Pick a random pair, then `batch_size` random windows of `timesteps` samples."""

    def __init__(
        self, pairs: list[Pair], timesteps: int, batch_size: int, rng: np.random.Generator
    ) -> None:
        for x, _ in pairs:
            if len(x) < timesteps:
                raise ValueError(f"audio of {len(x)} samples is shorter than {timesteps}")
        self._pairs = pairs
        self._timesteps = timesteps
        self._batch_size = batch_size
        self._rng = rng

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self._pairs[self._rng.integers(len(self._pairs))]
        offsets = self._rng.integers(len(x) - self._timesteps + 1, size=self._batch_size)
        bx = np.stack([x[o : o + self._timesteps] for o in offsets])
        by = np.stack([y[o : o + self._timesteps] for o in offsets])
        return torch.from_numpy(bx).unsqueeze(-1), torch.from_numpy(by).unsqueeze(-1)
