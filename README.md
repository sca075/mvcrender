# mvcrender

As per libraries like OpenCV are not available on Home Assistant's Python 3.13, this library provides
a fast C extension for manipulating images generated in Numpy.
This is a kind of collection of image processing utilities that can be used in Home Assistant.
The development focus is on improving the performance of the Valetudo Map Parsers Library.

## Functions Implemented
- AutoCrop: determines the bounding box of an image and crops it based on a background color.
- Blend: blend a color or image into another image based on a mask or alpha values.
- Draw: draw lines, circles, polygons, and polylines on an image.

## Supported platforms
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
