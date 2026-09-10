import numpy as np
import torch

from audio_effect_replicator.audio import SAMPLE_RATE
from audio_effect_replicator.evaluate import evaluate_prediction, time_inference
from audio_effect_replicator.model import FxReplicator


def test_evaluate_prediction_keys() -> None:
    x = np.random.default_rng(0).standard_normal(8192).astype(np.float32)
    result = evaluate_prediction(x, x)
    assert set(result) == {"mse", "esr", "lsd_db"}
    assert all(v == 0.0 for v in result.values())


def test_time_inference_reports_seconds_and_realtime_factor() -> None:
    model = FxReplicator(hidden=4)
    samples = np.zeros(SAMPLE_RATE // 2, np.float32)  # 0.5 s of audio
    result = time_inference(model, samples, 64, 16, 32, torch.device("cpu"))
    assert set(result) == {"inference_seconds", "realtime_factor"}
    assert result["inference_seconds"] > 0
    assert result["realtime_factor"] == 0.5 / result["inference_seconds"]
