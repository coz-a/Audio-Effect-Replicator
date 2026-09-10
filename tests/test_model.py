from pathlib import Path

import torch

from audio_effect_replicator.config import Config
from audio_effect_replicator.model import FxReplicator, load_checkpoint, save_checkpoint


def test_forward_shape() -> None:
    model = FxReplicator()
    y = model(torch.zeros(2, 32, 1))
    assert y.shape == (2, 32, 1)


def test_layer_sizes_match_2018() -> None:
    model = FxReplicator()
    assert (model.lstm1.input_size, model.lstm1.hidden_size) == (1, 64)
    assert (model.lstm2.input_size, model.lstm2.hidden_size) == (64, 64)
    assert (model.lstm_out.input_size, model.lstm_out.hidden_size) == (64, 1)


def test_checkpoint_roundtrip(tmp_path: Path, small_config: Config) -> None:
    model = FxReplicator()
    x = torch.randn(1, 16, 1)
    path = tmp_path / "model_000003.pt"
    save_checkpoint(path, model, small_config, epoch=3, val_loss=0.5)
    loaded, meta = load_checkpoint(path, torch.device("cpu"))
    torch.testing.assert_close(loaded(x), model(x))
    assert meta["epoch"] == 3
    assert meta["val_loss"] == 0.5
    assert meta["input_timesteps"] == small_config.input_timesteps
    assert meta["output_timesteps"] == small_config.output_timesteps
    assert not loaded.training
