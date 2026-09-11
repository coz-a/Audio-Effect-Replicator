from pathlib import Path

import pytest

from audio_effect_replicator.config import load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_default_config_is_the_2018_setup() -> None:
    c = load_config(CONFIGS / "default.yml")
    assert (c.sample_rate, c.input_timesteps, c.output_timesteps, c.batch_size) == (
        48000,
        5280,
        480,
        16,
    )


@pytest.mark.parametrize("device", ["ht1", "muff"])
def test_wright2019_configs(device: str) -> None:
    c = load_config(CONFIGS / f"wright2019-{device}.yml")
    assert c.sample_rate == 44100
    assert (c.input_timesteps, c.output_timesteps, c.batch_size, c.patience) == (5280, 480, 16, 50)
    data = Path("data/wright2019")
    assert c.train_data == [
        (data / "train" / f"{device}-input.wav", data / "train" / f"{device}-target.wav")
    ]
    assert c.val_data == [
        (data / "val" / f"{device}-input.wav", data / "val" / f"{device}-target.wav")
    ]
