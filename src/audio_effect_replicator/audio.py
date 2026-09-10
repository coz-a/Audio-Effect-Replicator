"""Read and write 48 kHz / mono / 16-bit PCM WAV files."""

import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48000
_INT16_MAX = 32767


class AudioFormatError(ValueError):
    """The WAV file is not 48 kHz / mono / 16-bit PCM."""


def load_wave(path: str | Path) -> np.ndarray:
    """Return the samples as float32 in [-1, 1]."""
    with wave.open(str(path), "rb") as w:
        channels, sampwidth, rate = w.getnchannels(), w.getsampwidth(), w.getframerate()
        if (channels, sampwidth, rate) != (1, 2, SAMPLE_RATE):
            raise AudioFormatError(
                f"{path}: expected mono 16-bit {SAMPLE_RATE} Hz, "
                f"got {channels} ch, {sampwidth * 8}-bit, {rate} Hz"
            )
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / _INT16_MAX


def save_wave(samples: np.ndarray, path: str | Path) -> None:
    """Write float samples (clipped to [-1, 1]) as mono 16-bit PCM."""
    pcm = np.round(np.clip(samples, -1.0, 1.0) * _INT16_MAX).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
