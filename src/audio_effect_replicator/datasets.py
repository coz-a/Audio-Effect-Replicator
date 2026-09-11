"""Download benchmark datasets described by the YAML manifests packaged next to this module."""

import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetFile:
    path: str
    url: str
    md5: str


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
    files = [DatasetFile(str(e["path"]), str(e["url"]), str(e["md5"])) for e in raw["files"]]
    return Manifest(
        name=str(raw["name"]),
        description=str(raw["description"]).strip(),
        license=str(raw["license"]),
        citation=str(raw["citation"]).strip(),
        sample_rate=int(raw["sample_rate"]),
        files=files,
    )


def fetch_dataset(manifest: Manifest, dest: Path) -> Path:
    """Download every file into `dest/<name>/`, skipping files whose MD5 already matches."""
    root = dest / manifest.name
    for entry in manifest.files:
        target = root / entry.path
        if target.exists() and _md5(target) == entry.md5:
            log.info("%s: already present", entry.path)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        log.info("%s: downloading", entry.path)
        urllib.request.urlretrieve(entry.url, target)
        if _md5(target) != entry.md5:
            target.unlink()
            raise RuntimeError(f"{entry.path}: checksum mismatch after download")
    return root


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
