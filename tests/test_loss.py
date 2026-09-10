import torch

from audio_effect_replicator.loss import tail_mse


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
