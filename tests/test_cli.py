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
