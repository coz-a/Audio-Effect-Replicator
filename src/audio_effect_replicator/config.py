"""Training / inference configuration loaded from YAML (same keys as the 2018 config.yml)."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """The YAML file is missing keys or holds inconsistent values."""


@dataclass(frozen=True)
class ModelSpec:
    type: str = "lstm2018"
    hidden: int = 64


@dataclass(frozen=True)
class LossSpec:
    type: str = "tail_mse"


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
    learning_rate: float = 1e-3
    model: ModelSpec = field(default_factory=ModelSpec)
    loss: LossSpec = field(default_factory=LossSpec)


_REQUIRED = (
    "input_timesteps",
    "output_timesteps",
    "batch_size",
    "max_epochs",
    "patience",
    "train_data",
    "val_data",
)

_OPTIONAL = ("steps_per_epoch", "validation_steps", "sample_rate", "learning_rate", "model", "loss")


def load_config(path: str | Path) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    missing = [key for key in _REQUIRED if key not in raw]
    if missing:
        raise ConfigError(f"{path}: missing keys: {', '.join(missing)}")
    unknown = [key for key in raw if key not in _REQUIRED and key not in _OPTIONAL]
    if unknown:
        raise ConfigError(f"{path}: unknown keys: {', '.join(sorted(unknown))}")
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
        learning_rate=float(raw.get("learning_rate", 1e-3)),
        model=_model_spec(raw.get("model")),
        loss=_loss_spec(raw.get("loss")),
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


def _model_spec(raw: object) -> ModelSpec:
    block = _block(raw, "model", ("type", "hidden"))
    return ModelSpec(type=str(block.get("type", "lstm2018")), hidden=int(block.get("hidden", 64)))


def _loss_spec(raw: object) -> LossSpec:
    return LossSpec(type=str(_block(raw, "loss", ("type",)).get("type", "tail_mse")))


def _block(raw: object, key: str, allowed: tuple[str, ...]) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{key} must be a mapping with keys {', '.join(allowed)}")
    unknown = [k for k in raw if k not in allowed]
    if unknown:
        raise ConfigError(f"{key}: unknown keys: {', '.join(sorted(unknown))}")
    return raw
