"""Download benchmark datasets described by the YAML manifests packaged next to this module.

Plain files are downloaded and MD5-checked in place. Archive entries (`extract: true`)
are downloaded, checked, extracted into a folder named after the archive and then
deleted; a `<archive>.md5` stamp records the verified checksum so later runs skip them.
Split zips (`name.z01`, `name.z02`, ..., `name.zip`) are extracted with 7-Zip (`7z`),
which reads multi-volume archives natively.
"""

import hashlib
import logging
import re
import shutil
import subprocess
import urllib.request
import zipfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

_PART = re.compile(r"\.z\d+$")


@dataclass(frozen=True)
class DatasetFile:
    path: str
    url: str
    md5: str
    extract: bool = False
    size: int = 0


@dataclass(frozen=True)
class Manifest:
    name: str
    description: str
    license: str
    citation: str
    sample_rate: int
    files: list[DatasetFile]


def list_datasets() -> list[str]:
    root = resources.files("audio_effect_replicator") / "datasets"
    return sorted(p.name[: -len(".yml")] for p in root.iterdir() if p.name.endswith(".yml"))


def packaged_manifest(name: str) -> Path:
    path = resources.files("audio_effect_replicator") / "datasets" / f"{name}.yml"
    if not path.is_file():
        raise ValueError(f"unknown dataset {name!r}; available: {', '.join(list_datasets())}")
    return Path(str(path))


def load_manifest(path: str | Path) -> Manifest:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    files = [
        DatasetFile(
            path=str(e["path"]),
            url=str(e["url"]),
            md5=str(e["md5"]),
            extract=bool(e.get("extract", False)),
            size=int(e.get("size", 0)),
        )
        for e in raw["files"]
    ]
    return Manifest(
        name=str(raw["name"]),
        description=str(raw["description"]).strip(),
        license=str(raw["license"]),
        citation=str(raw["citation"]).strip(),
        sample_rate=int(raw["sample_rate"]),
        files=files,
    )


def fetch_dataset(manifest: Manifest, dest: Path) -> Path:
    """Download every file into `dest/<name>/`, skipping files already verified."""
    root = dest / manifest.name
    for entry in manifest.files:
        target = root / entry.path
        if _already_done(entry, target):
            log.info("%s: already present", entry.path)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        log.info(
            "%s: downloading%s", entry.path, f" ({entry.size / 1e9:.1f} GB)" if entry.size else ""
        )
        urllib.request.urlretrieve(entry.url, target)
        if _md5(target) != entry.md5:
            target.unlink()
            raise RuntimeError(f"{entry.path}: checksum mismatch after download")
        if entry.extract:
            _stamp(target).write_text(entry.md5 + "\n")
            if target.suffix == ".zip":
                log.info("%s: extracting", entry.path)
                _extract(target)
    return root


def _already_done(entry: DatasetFile, target: Path) -> bool:
    if entry.extract:
        stamp = _stamp(target)
        return stamp.exists() and stamp.read_text().strip() == entry.md5
    return target.exists() and _md5(target) == entry.md5


def _stamp(target: Path) -> Path:
    return target.with_name(target.name + ".md5")


def _extract(archive: Path) -> None:
    """Extract `X.zip` (with its `X.z01`, `X.z02`, ... volumes) into `X/`, then delete them."""
    out_dir = archive.with_suffix("")
    parts = sorted(p for p in archive.parent.glob(f"{out_dir.name}.z*") if _PART.search(p.name))
    if parts:
        seven_zip = shutil.which("7z") or shutil.which("7zz")
        if seven_zip is None:
            raise RuntimeError(
                f"{archive.name} is a split archive; install 7-Zip (`7z`) to extract it"
            )
        subprocess.run(
            [seven_zip, "x", "-y", f"-o{out_dir}", str(archive)],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    else:
        with zipfile.ZipFile(archive) as z:
            z.extractall(out_dir)
    for p in [*parts, archive]:
        p.unlink(missing_ok=True)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
