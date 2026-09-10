import wave
from pathlib import Path

import numpy as np
import pytest

from audio_effect_replicator.audio import SAMPLE_RATE, AudioFormatError, load_wave, save_wave


def write_raw_wav(path: Path, channels: int, sampwidth: int, rate: int, frames: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(rate)
        w.writeframes(frames)


def test_roundtrip_preserves_samples(tmp_path: Path) -> None:
    samples = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
    path = tmp_path / "a.wav"
    save_wave(samples, path)
    loaded = load_wave(path)
    assert loaded.dtype == np.float32
    assert loaded.shape == samples.shape
    assert np.abs(loaded - samples).max() <= 1.0 / 32767


def test_save_clips_out_of_range(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    save_wave(np.array([2.0, -2.0], dtype=np.float32), path)
    loaded = load_wave(path)
    assert loaded[0] == pytest.approx(1.0)
    assert loaded[1] == pytest.approx(-1.0, abs=1.0 / 32767)


def test_load_reads_int16_little_endian(tmp_path: Path) -> None:
    path = tmp_path / "raw.wav"
    write_raw_wav(path, 1, 2, SAMPLE_RATE, np.array([0, 32767, -32767], dtype="<i2").tobytes())
    np.testing.assert_allclose(load_wave(path), [0.0, 1.0, -1.0])


@pytest.mark.parametrize(
    ("channels", "sampwidth", "rate"),
    [(2, 2, SAMPLE_RATE), (1, 1, SAMPLE_RATE), (1, 2, 44100)],
)
def test_load_rejects_wrong_format(
    tmp_path: Path, channels: int, sampwidth: int, rate: int
) -> None:
    path = tmp_path / "bad.wav"
    write_raw_wav(path, channels, sampwidth, rate, bytes(8 * channels * sampwidth))
    with pytest.raises(AudioFormatError):
        load_wave(path)
