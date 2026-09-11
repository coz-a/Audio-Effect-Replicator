import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile

from audio_effect_replicator.audio import SAMPLE_RATE, AudioFormatError, load_wave, save_wave

LSB = 1.0 / 32768


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
    assert np.abs(loaded - samples).max() <= LSB


def test_save_writes_16_bit_pcm_at_the_given_rate(tmp_path: Path) -> None:
    path = tmp_path / "r.wav"
    save_wave(np.zeros(10, np.float32), path, sample_rate=44100)
    info = soundfile.info(str(path))
    assert (info.samplerate, info.channels, info.subtype) == (44100, 1, "PCM_16")


def test_save_clips_out_of_range(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    save_wave(np.array([2.0, -2.0], dtype=np.float32), path)
    loaded = load_wave(path)
    assert loaded[0] == pytest.approx(1.0, abs=LSB)
    assert loaded[1] == pytest.approx(-1.0, abs=LSB)


def test_load_reads_int16_little_endian(tmp_path: Path) -> None:
    path = tmp_path / "raw.wav"
    write_raw_wav(path, 1, 2, SAMPLE_RATE, np.array([0, 32767, -32768], dtype="<i2").tobytes())
    np.testing.assert_allclose(load_wave(path), [0.0, 32767 / 32768, -1.0])


@pytest.mark.parametrize("subtype", ["PCM_24", "PCM_32", "FLOAT"])
def test_load_reads_other_subtypes(tmp_path: Path, subtype: str) -> None:
    samples = np.linspace(-0.5, 0.5, 100, dtype=np.float32)
    path = tmp_path / f"{subtype}.wav"
    soundfile.write(str(path), samples, SAMPLE_RATE, subtype=subtype)
    np.testing.assert_allclose(load_wave(path), samples, atol=LSB)


def test_load_accepts_matching_sample_rate(tmp_path: Path) -> None:
    path = tmp_path / "44k.wav"
    save_wave(np.zeros(10, np.float32), path, sample_rate=44100)
    assert load_wave(path, sample_rate=44100).shape == (10,)


@pytest.mark.parametrize(("channels", "rate"), [(2, SAMPLE_RATE), (1, 44100)])
def test_load_rejects_wrong_format(tmp_path: Path, channels: int, rate: int) -> None:
    path = tmp_path / "bad.wav"
    write_raw_wav(path, channels, 2, rate, bytes(16 * channels))
    with pytest.raises(AudioFormatError):
        load_wave(path)
