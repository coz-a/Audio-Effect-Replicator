import math

import numpy as np
import pytest

from audio_effect_replicator.metrics import esr, log_spectral_distance, mse, parameter_count
from audio_effect_replicator.model import FxReplicator


def signal(n: int = 8192) -> np.ndarray:
    t = np.arange(n) / 48000
    rng = np.random.default_rng(0)
    return (0.25 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(n)).astype(np.float32)


def test_identical_signals_score_zero() -> None:
    x = signal()
    assert mse(x, x) == 0.0
    assert esr(x, x) == 0.0
    assert log_spectral_distance(x, x) == 0.0


def test_mse_and_esr_known_values() -> None:
    target = np.array([1.0, -1.0, 2.0, 0.0], dtype=np.float32)
    zeros = np.zeros_like(target)
    assert mse(zeros, target) == pytest.approx(6.0 / 4)
    assert esr(zeros, target) == pytest.approx(1.0)


def test_lsd_of_scaled_signal_is_gain_in_db() -> None:
    x = signal()
    assert log_spectral_distance(2 * x, x) == pytest.approx(20 * math.log10(2), abs=0.05)


def test_length_mismatch_and_empty_are_rejected() -> None:
    x = signal()
    with pytest.raises(ValueError, match="length"):
        mse(x[:-1], x)
    with pytest.raises(ValueError, match="empty"):
        esr(x[:0], x[:0])


def test_esr_rejects_silent_target() -> None:
    with pytest.raises(ValueError, match="silent"):
        esr(np.ones(4, np.float32), np.zeros(4, np.float32))


def test_parameter_count_of_2018_model() -> None:
    # 4 gates x (in*h + h*h + 2h) per layer: (1,64) 17152 + (64,64) 33280 + (64,1) 268
    assert parameter_count(FxReplicator()) == 50_700


def test_lsd_is_insensitive_to_16_bit_quantization() -> None:
    x = signal()
    quantized = (np.round(x * 32768) / 32768).astype(np.float32)
    assert log_spectral_distance(x, quantized) == 0.0
