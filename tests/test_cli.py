import json
from pathlib import Path

import numpy as np
import pytest

from audio_effect_replicator import __version__
from audio_effect_replicator.audio import load_wave, save_wave
from audio_effect_replicator.cli import main


def train_on_cpu(config_file: Path, out_dir: Path) -> Path:
    args = ["train", "-c", str(config_file), "--device", "cpu", "--out-dir", str(out_dir)]
    assert main([*args, "--seed", "0"]) == 0
    return sorted((out_dir / "checkpoint").glob("*/model_*.pt"))[-1]


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_train_then_predict(tmp_path: Path, config_file: Path, wav_pair: tuple[Path, Path]) -> None:
    model = train_on_cpu(config_file, tmp_path / "run")
    x_path, _ = wav_pair
    out_path = tmp_path / "predicted.wav"
    args = ["predict", "-c", str(config_file), "-i", str(x_path), "-o", str(out_path)]
    assert main([*args, "-m", str(model), "--device", "cpu"]) == 0
    assert load_wave(out_path).shape == load_wave(x_path).shape


def test_predict_short_input(tmp_path: Path, config_file: Path) -> None:
    """A 5-sample file still yields a 5-sample output (the 2018 script would fail)."""
    model = train_on_cpu(config_file, tmp_path / "run")
    short = tmp_path / "short.wav"
    save_wave(np.zeros(5, np.float32), short)
    out_path = tmp_path / "short_out.wav"
    args = ["predict", "-c", str(config_file), "-i", str(short), "-o", str(out_path)]
    assert main([*args, "-m", str(model), "--device", "cpu"]) == 0
    assert load_wave(out_path).shape == (5,)


def test_predict_rejects_bad_wav(
    tmp_path: Path, config_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wav")
    args = ["predict", "-c", str(config_file), "-i", str(bad), "-o", str(tmp_path / "o.wav")]
    assert main([*args, "-m", str(tmp_path / "missing.pt")]) == 1
    assert "error:" in capsys.readouterr().err


def test_missing_config_is_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["train", "-c", str(tmp_path / "nope.yml")]) == 1
    assert "error:" in capsys.readouterr().err


def test_evaluate_prediction_path(
    tmp_path: Path, wav_pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    x_path, y_path = wav_pair
    out = tmp_path / "scores.json"
    args = ["evaluate", "-t", str(y_path), "--prediction", str(x_path), "--json", str(out)]
    assert main(args) == 0
    printed = capsys.readouterr().out
    assert "esr" in printed and "lsd_db" in printed
    scores = json.loads(out.read_text())
    assert set(scores) == {"prediction", "mse", "esr", "lsd_db"}
    assert scores["esr"] > 0


def test_evaluate_model_path(
    tmp_path: Path, config_file: Path, wav_pair: tuple[Path, Path]
) -> None:
    model = train_on_cpu(config_file, tmp_path / "run")
    x_path, y_path = wav_pair
    out = tmp_path / "scores.json"
    args = ["evaluate", "-t", str(y_path), "-m", str(model), "-i", str(x_path), "--device", "cpu"]
    assert main([*args, "--json", str(out)]) == 0
    scores = json.loads(out.read_text())
    assert {"mse", "esr", "lsd_db", "parameters", "inference_seconds", "realtime_factor"} <= set(
        scores
    )
    assert scores["device"] == "cpu"
    assert scores["parameters"] == 50_700


def test_evaluate_model_requires_input(
    tmp_path: Path, wav_pair: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    _, y_path = wav_pair
    assert main(["evaluate", "-t", str(y_path), "-m", str(tmp_path / "m.pt")]) == 1
    assert "input" in capsys.readouterr().err


def test_evaluate_rejects_model_and_prediction_together(wav_pair: tuple[Path, Path]) -> None:
    x_path, y_path = wav_pair
    with pytest.raises(SystemExit):
        main(["evaluate", "-t", str(y_path), "-m", "m.pt", "--prediction", str(x_path)])


def test_import_keras_then_evaluate(tmp_path: Path, wav_pair: tuple[Path, Path]) -> None:
    pytest.importorskip("h5py")
    from audio_effect_replicator.model import FxReplicator
    from tests.test_legacy import write_keras_h5

    h5 = tmp_path / "model_000031.h5"
    write_keras_h5(h5, FxReplicator(hidden=4))
    pt = tmp_path / "model.pt"
    args = ["import-keras", str(h5), "-o", str(pt), "--input-timesteps", "64"]
    assert main([*args, "--output-timesteps", "16"]) == 0
    x_path, y_path = wav_pair
    args = ["evaluate", "-t", str(y_path), "-m", str(pt), "-i", str(x_path), "--device", "cpu"]
    assert main(args) == 0
