import hashlib
import os
import shutil
import subprocess
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


def zip_manifest(tmp_path: Path) -> Path:
    """A manifest whose single entry is a zip archive holding Effect/test/x.wav."""
    import zipfile

    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    archive = src / "Effect.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("Effect/test/x.wav", b"wav" * 10)
        z.writestr("DRY/test/in.wav", b"dry" * 10)
    entry = {
        "path": "Effect.zip",
        "url": archive.as_uri(),
        "md5": hashlib.md5(archive.read_bytes()).hexdigest(),
        "extract": True,
    }
    manifest = tmp_path / "z.yml"
    raw = {
        "name": "zipped",
        "description": "d",
        "license": "L",
        "citation": "C",
        "sample_rate": 48000,
        "files": [entry],
    }
    manifest.write_text(yaml.safe_dump(raw))
    return manifest


def test_fetch_extracts_archives_into_a_folder_named_after_them(tmp_path: Path) -> None:
    m = load_manifest(zip_manifest(tmp_path))
    assert m.files[0].extract is True
    root = fetch_dataset(m, tmp_path / "data")
    assert (root / "Effect" / "Effect" / "test" / "x.wav").read_bytes() == b"wav" * 10
    assert (root / "Effect" / "DRY" / "test" / "in.wav").exists()
    assert not (root / "Effect.zip").exists(), "archive is removed after extraction"


def test_fetch_skips_extracted_archives_via_stamp(tmp_path: Path) -> None:
    m = load_manifest(zip_manifest(tmp_path))
    root = fetch_dataset(m, tmp_path / "data")
    stamp = root / "Effect.zip.md5"
    assert stamp.read_text().strip() == m.files[0].md5
    marker = root / "Effect" / "Effect" / "test" / "x.wav"
    before = marker.stat().st_mtime_ns
    fetch_dataset(m, tmp_path / "data")  # must not download or extract again
    assert marker.stat().st_mtime_ns == before


def test_plain_files_have_extract_false(tmp_path: Path) -> None:
    m = load_manifest(local_manifest(tmp_path))
    assert all(f.extract is False for f in m.files)


def test_packaged_tonetwist_manifests() -> None:
    names = [n for n in list_datasets() if n.startswith("tonetwist-afx-")]
    assert len(names) == 4
    total = 0
    for name in names:
        m = load_manifest(packaged_manifest(name))
        assert m.name == "tonetwist-afx"
        assert m.sample_rate == 48000
        assert m.license == "CC-BY-NC-4.0"
        assert all(f.extract for f in m.files)
        assert all(f.url.startswith("https://zenodo.org/records/") for f in m.files)
        total += len(m.files)
    assert total == 31 + 5 + 1 + 13


@pytest.mark.skipif(
    shutil.which("zip") is None, reason="needs the zip CLI to build a split archive"
)
def test_fetch_merges_split_archives(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    payload = src / "Split" / "big.bin"
    payload.parent.mkdir()
    payload.write_bytes(os.urandom(200_000))  # incompressible, > one 64 KB split
    subprocess.run(["zip", "-q", "-s", "64k", "-r", "Split.zip", "Split"], cwd=src, check=True)
    parts = sorted(src.glob("Split.z[0-9][0-9]")) + [src / "Split.zip"]
    assert len(parts) >= 3
    files = [
        {
            "path": p.name,
            "url": p.as_uri(),
            "md5": hashlib.md5(p.read_bytes()).hexdigest(),
            "extract": True,
        }
        for p in parts
    ]
    manifest = tmp_path / "s.yml"
    raw = {
        "name": "split",
        "description": "d",
        "license": "L",
        "citation": "C",
        "sample_rate": 48000,
        "files": files,
    }
    manifest.write_text(yaml.safe_dump(raw))
    root = fetch_dataset(load_manifest(manifest), tmp_path / "data")
    assert (root / "Split" / "Split" / "big.bin").read_bytes() == payload.read_bytes()
    leftovers = [p for p in root.glob("Split.z*") if p.suffix != ".md5"]
    assert not leftovers, "parts and archive are removed after extraction"


def test_fetch_reuses_a_verified_archive_without_downloading(tmp_path: Path) -> None:
    m = load_manifest(zip_manifest(tmp_path))
    root = tmp_path / "data" / "zipped"
    root.mkdir(parents=True)
    shutil.copy(tmp_path / "src" / "Effect.zip", root / "Effect.zip")
    (tmp_path / "src" / "Effect.zip").unlink()  # the "download" would now fail
    fetch_dataset(m, tmp_path / "data")
    assert (root / "Effect" / "Effect" / "test" / "x.wav").exists()
    assert (root / "Effect.zip.md5").read_text().strip() == m.files[0].md5
