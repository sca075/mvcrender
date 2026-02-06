# mvcrender

As per libraries like OpenCV are not available on Home Assistant's Python 3.12, this library provides
a fast C extension for manipulating images generated in Numpy.
This is a kind of collection of image processing utilities that can be used in Home Assistant.
The development focus is on improving the performance of the Valetudo Map Parsers Library.

## Functions Implemented
- **AutoCrop**: determines the bounding box of an image and crops it based on a background color.
- **Blend**: blend a color or image into another image based on a mask or alpha values.
- **Draw**: draw lines, circles, polygons, and polylines on an image.
- **Rooms**: process room masks, fill compressed pixels, extract points from masks.
- **Material**: render floor material patterns (wood planks, tiles) with ultra-fast C implementation.

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
from mvcrender.material import MaterialTileRenderer
from mvcrender.draw import line_u8, circle_u8
import numpy as np

# AutoCrop is a mixin you can use in your handler class
class Handler(AutoCrop):
    def __init__(self):
        AutoCrop.__init__(self, self)
        # set up any fields your handler expects (crop_area, etc.)

# Material rendering - ultra-fast floor patterns
tile = MaterialTileRenderer.get_tile("wood_horizontal", pixel_size=5)
image = np.zeros((500, 500, 4), dtype=np.uint8)
MaterialTileRenderer.apply_overlay_on_region(image, tile, 0, 500, 0, 500)

# Drawing primitives
line_u8(image, 0, 0, 100, 100, (255, 0, 0, 255), thickness=2)
circle_u8(image, 250, 250, 50, (0, 255, 0, 255), thickness=3)
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
