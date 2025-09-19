# mvcrender

High‑performance image autocropping for Home Assistant and Python 3.13+, with a fast C extension.

- Python: 3.13+
- Wheels: manylinux and musllinux (Alpine), macOS (x86_64, arm64), Windows (x86_64)
- Platforms: Linux x86_64/aarch64, macOS x86_64/arm64, Windows x86_64

## Install

```bash
pip install mvcrender
```

## Quick start

```python
from mvcrender.autocrop import AutoCrop

# AutoCrop is a mixin you can use in your handler class
class Handler(AutoCrop):
    def __init__(self):
        AutoCrop.__init__(self, self)
        # set up any fields your handler expects (crop_area, etc.)
```

## Why musllinux wheels?

Home Assistant often runs on Alpine-based Docker images. musllinux wheels ensure zero‑compile installs there.

## Development

- Build backend: setuptools (PEP 517)
- Source layout: src/
- C extension: src/mvcrender/autocrop/_native.c

### Local build (sdist + wheel)

```bash
python -m pip install -U build
python -m build
```

### Tests

A simple smoke example lives in tests/smoke.py. CI runs a minimal import test on the built wheels to keep the matrix fast and reliable.

## Release

- Tag a version like `v0.0.3` on GitHub
- GitHub Actions will build wheels for all targets (including musllinux) and upload them to PyPI using `PYPI_API_TOKEN`.

