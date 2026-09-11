import pytest
import torch

from audio_effect_replicator.config import LossSpec
from audio_effect_replicator.loss import build_loss, esr, esr_preemphasis, tail_mse


def test_tail_mse_ignores_warmup_region() -> None:
    y_true = torch.zeros(2, 10, 1)
    y_pred = torch.zeros(2, 10, 1)
    y_pred[:, :6] = 100.0  # warm-up region, must be ignored
    y_pred[:, 6:] = 2.0
    assert tail_mse(y_pred, y_true, timesteps=4).item() == 4.0


def test_tail_mse_is_scalar() -> None:
    loss = tail_mse(torch.ones(1, 5, 1), torch.zeros(1, 5, 1), timesteps=5)
    assert loss.shape == ()
    assert loss.item() == 1.0


def test_esr_of_a_silent_prediction_is_one() -> None:
    target = torch.randn(2, 10, 1)
    assert esr(torch.zeros_like(target), target, timesteps=10).item() == pytest.approx(1.0)


def test_esr_ignores_the_warmup_region() -> None:
    target = torch.ones(1, 10, 1)
    pred = target.clone()
    pred[:, :6] = 100.0
    assert esr(pred, target, timesteps=4).item() == pytest.approx(0.0)


def test_esr_rejects_a_silent_target() -> None:
    with pytest.raises(ValueError, match="silent"):
        esr(torch.ones(1, 4, 1), torch.zeros(1, 4, 1), timesteps=4)


def test_preemphasis_weights_high_frequency_error_more() -> None:
    """Two errors of equal energy score alike under ESR; the faster one costs more here."""
    t = torch.linspace(0, 100, 400)
    target = torch.sin(t).reshape(1, 400, 1)
    slow = target + 0.05 * torch.sin(t / 10).reshape(1, 400, 1)
    fast = target + 0.05 * torch.sin(t * 3).reshape(1, 400, 1)
    assert esr(slow, target, 400).item() == pytest.approx(esr(fast, target, 400).item(), rel=0.1)
    assert esr_preemphasis(fast, target, 400) > 5 * esr_preemphasis(slow, target, 400)


def test_preemphasis_keeps_penalising_a_dc_offset() -> None:
    """The high-pass alone would nearly ignore a constant offset; the DC term restores it."""
    target = torch.sin(torch.linspace(0, 20, 100)).reshape(1, 100, 1)
    dc_term = 0.1**2 / torch.mean(target**2).item()
    assert esr_preemphasis(target + 0.1, target, 100).item() >= dc_term


def test_build_loss_returns_the_named_loss() -> None:
    target, pred = torch.ones(1, 4, 1), torch.zeros(1, 4, 1)
    assert build_loss(LossSpec())(pred, target, 4) == tail_mse(pred, target, 4)
    assert build_loss(LossSpec(type="esr"))(pred, target, 4) == esr(pred, target, 4)


def test_build_loss_rejects_an_unknown_type() -> None:
    with pytest.raises(ValueError, match="tail_mse"):
        build_loss(LossSpec(type="nope"))
