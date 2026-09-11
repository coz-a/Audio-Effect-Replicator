from pathlib import Path

import pytest
import torch

from audio_effect_replicator.config import ModelSpec
from audio_effect_replicator.model import (
    FxReplicator,
    SkipLstm,
    WrightLstm,
    build_model,
    load_checkpoint,
    save_checkpoint,
)


def test_forward_shape() -> None:
    model = FxReplicator()
    y = model(torch.zeros(2, 32, 1))
    assert y.shape == (2, 32, 1)


def test_layer_sizes_match_2018() -> None:
    model = FxReplicator()
    assert (model.lstm1.input_size, model.lstm1.hidden_size) == (1, 64)
    assert (model.lstm2.input_size, model.lstm2.hidden_size) == (64, 64)
    assert (model.lstm_out.input_size, model.lstm_out.hidden_size) == (64, 1)


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = FxReplicator()
    x = torch.randn(1, 16, 1)
    path = tmp_path / "model_000003.pt"
    save_checkpoint(path, model, 64, 16, epoch=3, val_loss=0.5)
    loaded, meta = load_checkpoint(path, torch.device("cpu"))
    torch.testing.assert_close(loaded(x), model(x))
    assert meta["epoch"] == 3
    assert meta["val_loss"] == 0.5
    assert meta["input_timesteps"] == 64
    assert meta["output_timesteps"] == 16
    assert not loaded.training
    assert meta["sample_rate"] == 48000


def test_init_matches_keras_2_1() -> None:
    """Keras 2.1 LSTM defaults: orthogonal recurrent kernels, forget-gate bias 1, other biases 0."""
    model = FxReplicator()
    for lstm in (model.lstm1, model.lstm2, model.lstm_out):
        h = lstm.hidden_size
        _, weight_hh, bias_ih, bias_hh = lstm.all_weights[0]
        for gate in range(4):
            block = weight_hh[gate * h : (gate + 1) * h]
            torch.testing.assert_close(block @ block.T, torch.eye(h), atol=1e-5, rtol=0)
        expected_bias = torch.zeros(4 * h)
        expected_bias[h : 2 * h] = 1.0  # gate order in torch: input, forget, cell, output
        torch.testing.assert_close(bias_ih, expected_bias)
        torch.testing.assert_close(bias_hh, torch.zeros(4 * h))


def test_init_input_weights_are_glorot_bounded() -> None:
    model = FxReplicator()
    w = model.lstm1.all_weights[0][0]  # weight_ih, shape (4*64, 1): fan_in 1, fan_out 256
    bound = (6.0 / (1 + 4 * 64)) ** 0.5
    assert w.abs().max() <= bound
    assert w.abs().max() > 0.5 * bound  # not the torch default (+-1/sqrt(64) = 0.125)


def test_checkpoint_stores_sample_rate(tmp_path: Path) -> None:
    path = tmp_path / "m.pt"
    model = FxReplicator(hidden=4)
    save_checkpoint(path, model, 64, 16, epoch=1, val_loss=0.1, sample_rate=44100)
    assert load_checkpoint(path, torch.device("cpu"))[1]["sample_rate"] == 44100


def test_old_checkpoint_without_sample_rate_defaults_to_48000(tmp_path: Path) -> None:
    model = FxReplicator(hidden=4)
    path = tmp_path / "old.pt"
    torch.save(
        {
            "version": 1,
            "hidden": 4,
            "input_timesteps": 64,
            "output_timesteps": 16,
            "epoch": 1,
            "val_loss": 0.1,
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    assert load_checkpoint(path, torch.device("cpu"))[1]["sample_rate"] == 48000


def test_wright_lstm_shape_and_size() -> None:
    model = WrightLstm(hidden=64)
    assert model(torch.zeros(2, 32, 1)).shape == (2, 32, 1)
    # one LSTM layer 4*(1*64 + 64*64 + 2*64) = 17152, plus a 64 -> 1 linear head
    assert sum(p.numel() for p in model.parameters()) == 17_217


def test_skip_lstm_adds_the_input() -> None:
    model = SkipLstm(hidden=8)
    with torch.no_grad():
        model.inner.head.weight.zero_()
        model.inner.head.bias.zero_()
    x = torch.randn(2, 16, 1)
    torch.testing.assert_close(model(x), x)


def test_build_model_returns_each_architecture() -> None:
    assert isinstance(build_model(ModelSpec()), FxReplicator)
    assert isinstance(build_model(ModelSpec(type="wright", hidden=32)), WrightLstm)
    assert isinstance(build_model(ModelSpec(type="skip", hidden=32)), SkipLstm)
    assert build_model(ModelSpec(type="wright", hidden=32)).hidden == 32


def test_build_model_rejects_an_unknown_type() -> None:
    with pytest.raises(ValueError, match="lstm2018"):
        build_model(ModelSpec(type="nope"))
