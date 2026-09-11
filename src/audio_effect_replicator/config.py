"""Training / inference configuration loaded from YAML (same keys as the 2018 config.yml)."""

from dataclasses import dataclass
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """The YAML file is missing keys or holds inconsistent values."""


@dataclass(frozen=True)
class Config:
    input_timesteps: int
    output_timesteps: int
    batch_size: int
    max_epochs: int
    patience: int
    train_data: list[tuple[Path, Path]]
    val_data: list[tuple[Path, Path]]
    steps_per_epoch: int = 100
    validation_steps: int = 10
    sample_rate: int = 48000


_REQUIRED = (
    "input_timesteps",
    "output_timesteps",
    "batch_size",
    "max_epochs",
    "patience",
    "train_data",
    "val_data",
)


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    missing = [key for key in _REQUIRED if key not in raw]
    if missing:
        raise ConfigError(f"{path}: missing keys: {', '.join(missing)}")
    config = Config(
        input_timesteps=int(raw["input_timesteps"]),
        output_timesteps=int(raw["output_timesteps"]),
        batch_size=int(raw["batch_size"]),
        max_epochs=int(raw["max_epochs"]),
        patience=int(raw["patience"]),
        train_data=_pairs(raw["train_data"], "train_data"),
        val_data=_pairs(raw["val_data"], "val_data"),
        steps_per_epoch=int(raw.get("steps_per_epoch", 100)),
        validation_steps=int(raw.get("validation_steps", 10)),
        sample_rate=int(raw.get("sample_rate", 48000)),
    )
    if config.output_timesteps > config.input_timesteps:
        raise ConfigError("output_timesteps must not exceed input_timesteps")
    return config


def _pairs(items: object, key: str) -> list[tuple[Path, Path]]:
    if not isinstance(items, list) or not items:
        raise ConfigError(f"{key} must be a non-empty list of [input, output] pairs")
    pairs: list[tuple[Path, Path]] = []
    for item in items:
        if not isinstance(item, list) or len(item) != 2:
            raise ConfigError(f"{key} entries must be [input, output] pairs")
        pairs.append((Path(item[0]), Path(item[1])))
    return pairs
