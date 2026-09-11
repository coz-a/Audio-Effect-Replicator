import math
from pathlib import Path

import pytest
import torch

from audio_effect_replicator.config import Config
from audio_effect_replicator.model import load_checkpoint
from audio_effect_replicator.train import train


def test_train_writes_checkpoint_and_logs(tmp_path: Path, small_config: Config) -> None:
    ckpt_dir = train(small_config, torch.device("cpu"), out_dir=tmp_path, seed=0)

    assert ckpt_dir.parent == tmp_path / "checkpoint"
    checkpoints = sorted(ckpt_dir.glob("model_*.pt"))
    assert checkpoints, "no checkpoint was written"
    assert checkpoints[0].name == "model_000001.pt"

    _, meta = load_checkpoint(checkpoints[-1], torch.device("cpu"))
    assert math.isfinite(meta["val_loss"])
    assert 1 <= meta["epoch"] <= small_config.max_epochs

    tb_dir = tmp_path / "tensorboard" / ckpt_dir.name
    assert any(tb_dir.iterdir()), "no TensorBoard event file"


def test_train_stops_within_max_epochs(tmp_path: Path, small_config: Config) -> None:
    ckpt_dir = train(small_config, torch.device("cpu"), out_dir=tmp_path, seed=1)
    epochs = [int(p.stem.split("_")[1]) for p in ckpt_dir.glob("model_*.pt")]
    assert max(epochs) <= small_config.max_epochs


def test_gradient_clipping_keeps_a_collapsing_run_alive(
    tmp_path: Path, small_config: Config
) -> None:
    """A single huge gradient step can zero the model out; clipping must bound the update."""
    from dataclasses import replace

    import torch.nn as nn

    from audio_effect_replicator.train import _clip_gradients

    model = nn.Linear(1, 1)
    model.weight.grad = torch.full_like(model.weight, 1000.0)
    model.bias.grad = torch.full_like(model.bias, 1000.0)
    _clip_gradients(model, replace(small_config, grad_clip=1.0))
    squares = [p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None]
    total = torch.sqrt(torch.stack(squares).sum())
    assert total.item() == pytest.approx(1.0, abs=1e-5)


def test_gradients_are_untouched_when_clipping_is_off(small_config: Config) -> None:
    import torch.nn as nn

    from audio_effect_replicator.train import _clip_gradients

    model = nn.Linear(1, 1)
    model.weight.grad = torch.full_like(model.weight, 1000.0)
    _clip_gradients(model, small_config)  # grad_clip defaults to 0
    assert model.weight.grad.item() == 1000.0
