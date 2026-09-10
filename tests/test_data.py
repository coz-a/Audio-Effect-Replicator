from pathlib import Path

import numpy as np
import pytest
import torch

from audio_effect_replicator.audio import save_wave
from audio_effect_replicator.data import WindowSampler, load_pairs


def test_load_pairs_returns_float_arrays(wav_pair: tuple[Path, Path]) -> None:
    pairs = load_pairs([wav_pair])
    assert len(pairs) == 1
    x, y = pairs[0]
    assert x.dtype == np.float32 and y.dtype == np.float32
    assert x.shape == y.shape


def test_load_pairs_rejects_length_mismatch(tmp_path: Path) -> None:
    save_wave(np.zeros(10, np.float32), tmp_path / "x.wav")
    save_wave(np.zeros(11, np.float32), tmp_path / "y.wav")
    with pytest.raises(ValueError, match="length"):
        load_pairs([(tmp_path / "x.wav", tmp_path / "y.wav")])


def test_sample_shapes_and_alignment() -> None:
    x = np.arange(100, dtype=np.float32)
    pairs = [(x, x + 1000)]
    sampler = WindowSampler(pairs, timesteps=8, batch_size=3, rng=np.random.default_rng(0))
    bx, by = sampler.sample()
    assert bx.shape == (3, 8, 1) and by.shape == (3, 8, 1)
    assert bx.dtype == torch.float32
    # y window is cut at the same offset as x
    torch.testing.assert_close(by, bx + 1000)
    # each window is a contiguous run
    assert torch.all(bx[:, 1:, 0] - bx[:, :-1, 0] == 1)


def test_sample_is_reproducible_with_seed() -> None:
    x = np.arange(100, dtype=np.float32)
    a = WindowSampler([(x, x)], 8, 2, np.random.default_rng(7)).sample()[0]
    b = WindowSampler([(x, x)], 8, 2, np.random.default_rng(7)).sample()[0]
    torch.testing.assert_close(a, b)


def test_too_short_audio_is_rejected() -> None:
    x = np.zeros(5, np.float32)
    with pytest.raises(ValueError, match="shorter"):
        WindowSampler([(x, x)], timesteps=8, batch_size=1, rng=np.random.default_rng(0))
