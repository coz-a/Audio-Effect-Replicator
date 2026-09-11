"""Read mono WAV files of any PCM / float subtype; write 16-bit PCM."""

from pathlib import Path

import numpy as np
import soundfile

SAMPLE_RATE = 48000


class AudioFormatError(ValueError):
    """The WAV file is not mono or not at the expected sample rate."""


def load_wave(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return the samples as float32 in [-1, 1]."""
    samples, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    channels = samples.shape[1]
    if channels != 1 or rate != sample_rate:
        raise AudioFormatError(
            f"{path}: expected mono {sample_rate} Hz, got {channels} ch, {rate} Hz"
        )
    return np.ascontiguousarray(samples[:, 0])


def save_wave(samples: np.ndarray, path: str | Path, sample_rate: int = SAMPLE_RATE) -> None:
    """Write float samples (clipped to [-1, 1]) as mono 16-bit PCM."""
    clipped = np.clip(samples, -1.0, 1.0).astype(np.float32)
    soundfile.write(str(path), clipped, sample_rate, subtype="PCM_16")
