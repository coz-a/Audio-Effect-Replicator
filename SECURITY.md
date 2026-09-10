# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| `main` and the latest release (`v1.0-modernized` or newer) | Yes |
| `2018-original` tag | No |

The `2018-original` tag preserves the February 2018 implementation exactly as
it was released, including its `requirements.txt` (TensorFlow GPU 1.5,
Keras 2.1 and other packages with known vulnerabilities). It is kept for
historical reference only and must not be used in production or
security-sensitive environments. Issues about running the 2018 code are not
supported.

Dependencies on `main` are pinned in `uv.lock` and updated through Dependabot.

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/coz-a/Audio-Effect-Replicator/security/advisories/new)
rather than in a public issue. You will get a reply within 14 days.
