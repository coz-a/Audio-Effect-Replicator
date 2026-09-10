import math
from pathlib import Path

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
