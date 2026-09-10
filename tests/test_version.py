from audio_effect_replicator import __version__


def test_version_is_semver() -> None:
    major, minor, patch = __version__.split(".")
    assert all(part.isdigit() for part in (major, minor, patch))
