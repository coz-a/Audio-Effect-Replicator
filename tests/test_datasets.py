import hashlib
from pathlib import Path

import pytest
import yaml

from audio_effect_replicator.datasets import (
    Manifest,
    fetch_dataset,
    list_datasets,
    load_manifest,
    packaged_manifest,
)


def local_manifest(tmp_path: Path, corrupt: bool = False) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    files = []
    for name in ("a.wav", "sub/b.wav"):
        p = src / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(name.encode() * 100)
        md5 = "0" * 32 if corrupt else hashlib.md5(p.read_bytes()).hexdigest()
        files.append({"path": name, "url": p.as_uri(), "md5": md5})
    manifest = tmp_path / "m.yml"
    raw = {
        "name": "toy",
        "description": "d",
        "license": "L",
        "citation": "C",
        "sample_rate": 44100,
        "files": files,
    }
    manifest.write_text(yaml.safe_dump(raw))
    return manifest


def test_load_manifest(tmp_path: Path) -> None:
    m = load_manifest(local_manifest(tmp_path))
    assert isinstance(m, Manifest)
    assert (m.name, m.sample_rate, len(m.files)) == ("toy", 44100, 2)
    assert m.files[1].path == "sub/b.wav"


def test_fetch_downloads_and_verifies(tmp_path: Path) -> None:
    m = load_manifest(local_manifest(tmp_path))
    root = fetch_dataset(m, tmp_path / "data")
    assert root == tmp_path / "data" / "toy"
    assert (root / "sub" / "b.wav").read_bytes() == b"sub/b.wav" * 100


def test_fetch_skips_files_that_already_match(tmp_path: Path) -> None:
    m = load_manifest(local_manifest(tmp_path))
    root = fetch_dataset(m, tmp_path / "data")
    before = (root / "a.wav").stat().st_mtime_ns
    fetch_dataset(m, tmp_path / "data")
    assert (root / "a.wav").stat().st_mtime_ns == before


def test_fetch_rejects_checksum_mismatch(tmp_path: Path) -> None:
    m = load_manifest(local_manifest(tmp_path, corrupt=True))
    with pytest.raises(RuntimeError, match="checksum"):
        fetch_dataset(m, tmp_path / "data")
    assert not (tmp_path / "data" / "toy" / "a.wav").exists()


def test_packaged_wright2019_manifest() -> None:
    assert "wright2019" in list_datasets()
    m = load_manifest(packaged_manifest("wright2019"))
    assert m.sample_rate == 44100
    assert m.license == "CC-BY-NC-4.0"
    assert len(m.files) == 12
    assert {f.path.split("/")[0] for f in m.files} == {"train", "val", "test"}
    assert all(f.url.startswith("https://raw.githubusercontent.com/") for f in m.files)
    assert all(len(f.md5) == 32 for f in m.files)


def test_unknown_dataset_name() -> None:
    with pytest.raises(ValueError, match="wright2019"):
        packaged_manifest("nope")
