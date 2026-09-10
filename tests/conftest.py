from pathlib import Path

import numpy as np
import pytest
import yaml

from audio_effect_replicator.audio import SAMPLE_RATE, save_wave
from audio_effect_replicator.config import Config, load_config


@pytest.fixture
def wav_pair(tmp_path: Path) -> tuple[Path, Path]:
    """Two seconds of a sine with noise (x) and its tanh distortion (y)."""
    rng = np.random.default_rng(0)
    t = np.arange(2 * SAMPLE_RATE) / SAMPLE_RATE
    x = 0.5 * np.sin(2 * np.pi * 220 * t) + 0.01 * rng.standard_normal(t.size)
    y = 0.8 * np.tanh(3 * x)
    x_path, y_path = tmp_path / "x.wav", tmp_path / "y.wav"
    save_wave(x.astype(np.float32), x_path)
    save_wave(y.astype(np.float32), y_path)
    return x_path, y_path


@pytest.fixture
def config_file(tmp_path: Path, wav_pair: tuple[Path, Path]) -> Path:
    x_path, y_path = wav_pair
    raw = {
        "input_timesteps": 64,
        "output_timesteps": 16,
        "batch_size": 4,
        "max_epochs": 3,
        "patience": 1,
        "steps_per_epoch": 2,
        "validation_steps": 1,
        "train_data": [[str(x_path), str(y_path)]],
        "val_data": [[str(x_path), str(y_path)]],
    }
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.fixture
def small_config(config_file: Path) -> Config:
    return load_config(config_file)
