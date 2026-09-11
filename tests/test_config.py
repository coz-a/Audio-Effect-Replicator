from pathlib import Path

import pytest

from audio_effect_replicator.config import ConfigError, load_config


def test_load_config_reads_all_keys(config_file: Path, wav_pair: tuple[Path, Path]) -> None:
    config = load_config(config_file)
    assert config.input_timesteps == 64
    assert config.output_timesteps == 16
    assert config.batch_size == 4
    assert config.max_epochs == 3
    assert config.patience == 1
    assert config.steps_per_epoch == 2
    assert config.validation_steps == 1
    assert config.train_data == [wav_pair]
    assert config.val_data == [wav_pair]


def test_steps_default_to_2018_values(tmp_path: Path) -> None:
    path = tmp_path / "c.yml"
    path.write_text(
        "input_timesteps: 5280\noutput_timesteps: 480\nbatch_size: 16\n"
        "max_epochs: 10000\npatience: 50\n"
        "train_data:\n  - [a.wav, b.wav]\nval_data:\n  - [c.wav, d.wav]\n"
    )
    config = load_config(path)
    assert (config.steps_per_epoch, config.validation_steps) == (100, 10)
    assert config.train_data == [(Path("a.wav"), Path("b.wav"))]


def test_missing_key_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "c.yml"
    path.write_text("input_timesteps: 10\n")
    with pytest.raises(ConfigError, match="output_timesteps"):
        load_config(path)


def test_output_longer_than_input_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "c.yml"
    path.write_text(
        "input_timesteps: 10\noutput_timesteps: 20\nbatch_size: 1\nmax_epochs: 1\n"
        "patience: 1\ntrain_data:\n  - [a.wav, b.wav]\nval_data:\n  - [a.wav, b.wav]\n"
    )
    with pytest.raises(ConfigError, match="output_timesteps"):
        load_config(path)


def test_empty_data_list_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "c.yml"
    path.write_text(
        "input_timesteps: 10\noutput_timesteps: 5\nbatch_size: 1\nmax_epochs: 1\n"
        "patience: 1\ntrain_data: []\nval_data:\n  - [a.wav, b.wav]\n"
    )
    with pytest.raises(ConfigError, match="train_data"):
        load_config(path)


def test_non_mapping_top_level_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "c.yml"
    path.write_text("- just\n- a list\n")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(path)


def test_non_pair_entry_is_config_error(tmp_path: Path) -> None:
    path = tmp_path / "c.yml"
    path.write_text(
        "input_timesteps: 10\noutput_timesteps: 5\nbatch_size: 1\nmax_epochs: 1\n"
        "patience: 1\ntrain_data:\n  - [only_one.wav]\nval_data:\n  - [a.wav, b.wav]\n"
    )
    with pytest.raises(ConfigError, match="pairs"):
        load_config(path)


def test_sample_rate_defaults_to_48000(config_file: Path) -> None:
    assert load_config(config_file).sample_rate == 48000


def test_sample_rate_is_read(tmp_path: Path) -> None:
    path = tmp_path / "c.yml"
    path.write_text(
        "sample_rate: 44100\ninput_timesteps: 10\noutput_timesteps: 5\nbatch_size: 1\n"
        "max_epochs: 1\npatience: 1\n"
        "train_data:\n  - [a.wav, b.wav]\nval_data:\n  - [a.wav, b.wav]\n"
    )
    assert load_config(path).sample_rate == 44100
