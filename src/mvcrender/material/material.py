"""
Material patterns rendered as small RGBA tiles.

Provides fast rendering of floor materials like wood planks and tiles
using optimized C implementations for maximum performance.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Final, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from . import _native
    HAS_NATIVE = True
except ImportError:
    HAS_NATIVE = False
    from mvcrender.draw import line_u8

# Type aliases
Color = Tuple[int, int, int, int]  # RGBA
NumpyArray = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class _MaterialSpec:
    """Specification for a material pattern."""
    cells: int
    kind: str  # "wood_h", "wood_v", "tile"


class MaterialTileRenderer:
    """
    Material patterns rendered as small RGBA tiles.

    Wood is drawn as staggered rectangular planks (brick-like) with ONLY thin seams
    (no extra inner grain line). Tiles are drawn as simple grid patterns.

    Usage:
        # Get a tile pattern
        tile = MaterialTileRenderer.get_tile("wood_horizontal", pixel_size=5)
        
        # Apply to an image region
        MaterialTileRenderer.apply_overlay_on_region(image, tile, r0, r1, c0, c1)
        
        # Customize colors
        MaterialTileRenderer.set_colors(
            wood_rgba=(50, 50, 50, 40),
            tile_rgba=(30, 30, 30, 50)
        )
    """

    # Default neutral contour colors (RGBA) – keep subtle
    _DEFAULT_WOOD_RGBA: Final[Color] = (40, 40, 40, 38)
    _DEFAULT_TILE_RGBA: Final[Color] = (40, 40, 40, 45)

    # Current colors (can be updated via set_colors)
    WOOD_RGBA: Color = _DEFAULT_WOOD_RGBA
    TILE_RGBA: Color = _DEFAULT_TILE_RGBA

    _SPECS: Final[dict[str, _MaterialSpec]] = {
        "wood_horizontal": _MaterialSpec(cells=36, kind="wood_h"),
        "wood_vertical": _MaterialSpec(cells=36, kind="wood_v"),
        "tile": _MaterialSpec(cells=4, kind="tile"),
    }

    @classmethod
    def set_colors(
        cls, wood_rgba: Optional[Color] = None, tile_rgba: Optional[Color] = None
    ) -> None:
        """
        Update the material colors used for rendering.

        Args:
            wood_rgba: RGBA color for wood lines (default: use current)
            tile_rgba: RGBA color for tile lines (default: use current)
        """
        if wood_rgba is not None:
            cls.WOOD_RGBA = wood_rgba
        if tile_rgba is not None:
            cls.TILE_RGBA = tile_rgba

    @classmethod
    def reset_colors(cls) -> None:
        """Reset material colors to defaults."""
        cls.WOOD_RGBA = cls._DEFAULT_WOOD_RGBA
        cls.TILE_RGBA = cls._DEFAULT_TILE_RGBA

    @staticmethod
    def _empty_rgba(h: int, w: int) -> NumpyArray:
        """Create an empty RGBA array."""
        return np.zeros((h, w, 4), dtype=np.uint8)

    @staticmethod
    def _thin_px(pixel_size: int) -> int:
        """Thin seam thickness in pixels (for pixel_size 5/7 -> 1 px)."""
        return 1 if pixel_size <= 7 else 2

    @staticmethod
    def _draw_rect_outline(
        tile: NumpyArray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        thickness: int,
        rgba: Color,
    ) -> None:
        """Draw rectangle outline using mvcrender.line_u8."""
        if x1 <= x0 or y1 <= y0:
            return

        # Draw four lines to form rectangle outline
        # Top line
        line_u8(tile, x0, y0, x1 - 1, y0, rgba, thickness)
        # Bottom line
        line_u8(tile, x0, y1 - 1, x1 - 1, y1 - 1, rgba, thickness)
        # Left line
        line_u8(tile, x0, y0, x0, y1 - 1, rgba, thickness)
        # Right line
        line_u8(tile, x1 - 1, y0, x1 - 1, y1 - 1, rgba, thickness)

    @staticmethod
    def _wood_planks_horizontal(tile_px: int, pixel_size: int) -> NumpyArray:
        """
        Horizontal wood planks as staggered rectangles.
        ONLY thin seams (no inner lines).
        """
        t = MaterialTileRenderer._empty_rgba(tile_px, tile_px)
        seam = MaterialTileRenderer._thin_px(pixel_size)

        # Plank size in CELLS (tweak here)
        plank_h_cells = 3
        plank_w_cells = 24  # longer planks -> looks less like tiles

        plank_h = plank_h_cells * pixel_size
        plank_w = plank_w_cells * pixel_size

        rows = max(1, tile_px // plank_h)
        cols = max(1, (tile_px + plank_w - 1) // plank_w)

        for r in range(rows + 1):
            y0 = r * plank_h
            y1 = y0 + plank_h
            offset = (plank_w // 2) if (r % 2 == 1) else 0

            for c in range(cols + 1):
                x0 = c * plank_w - offset
                x1 = x0 + plank_w

                cx0 = max(0, x0)
                cy0 = max(0, y0)
                cx1 = min(tile_px, x1)
                cy1 = min(tile_px, y1)

                MaterialTileRenderer._draw_rect_outline(
                    t, cx0, cy0, cx1, cy1, seam, MaterialTileRenderer.WOOD_RGBA
                )

        return t

    @staticmethod
    def _wood_planks_vertical(tile_px: int, pixel_size: int) -> NumpyArray:
        """Vertical wood planks as staggered rectangles, ONLY thin seams."""
        t = MaterialTileRenderer._empty_rgba(tile_px, tile_px)
        seam = MaterialTileRenderer._thin_px(pixel_size)

        plank_w_cells = 3
        plank_h_cells = 24

        plank_w = plank_w_cells * pixel_size
        plank_h = plank_h_cells * pixel_size

        cols = max(1, tile_px // plank_w)
        rows = max(1, (tile_px + plank_h - 1) // plank_h)

        for c in range(cols + 1):
            x0 = c * plank_w
            x1 = x0 + plank_w
            offset = (plank_h // 2) if (c % 2 == 1) else 0

            for r in range(rows + 1):
                y0 = r * plank_h - offset
                y1 = y0 + plank_h

                cx0 = max(0, x0)
                cy0 = max(0, y0)
                cx1 = min(tile_px, x1)
                cy1 = min(tile_px, y1)

                MaterialTileRenderer._draw_rect_outline(
                    t, cx0, cy0, cx1, cy1, seam, MaterialTileRenderer.WOOD_RGBA
                )

        return t

    @staticmethod
    def _tile_pixels(cells: int, pixel_size: int) -> NumpyArray:
        """Draw tile grid using mvcrender.line_u8."""
        size = cells * pixel_size
        t = MaterialTileRenderer._empty_rgba(size, size)
        th = MaterialTileRenderer._thin_px(pixel_size)
        rgba = MaterialTileRenderer.TILE_RGBA

        # Draw horizontal line at top
        line_u8(t, 0, 0, size - 1, 0, rgba, th)
        # Draw vertical line at left
        line_u8(t, 0, 0, 0, size - 1, rgba, th)

        return t

    @staticmethod
    @lru_cache(maxsize=64)
    def get_tile(material: str, pixel_size: int) -> Optional[NumpyArray]:
        """
        Get a material tile pattern.

        Args:
            material: Material type ("wood_horizontal", "wood_vertical", "tile")
            pixel_size: Size of each pixel/cell in the pattern

        Returns:
            RGBA numpy array with the tile pattern, or None if invalid
        """
        spec = MaterialTileRenderer._SPECS.get(material)
        if spec is None or pixel_size <= 0:
            return None

        # Use fast C implementation if available
        if HAS_NATIVE:
            wood_r, wood_g, wood_b, wood_a = MaterialTileRenderer.WOOD_RGBA
            tile_r, tile_g, tile_b, tile_a = MaterialTileRenderer.TILE_RGBA

            if spec.kind == "tile":
                return _native.generate_tile_pattern(
                    spec.cells, pixel_size, tile_r, tile_g, tile_b, tile_a
                )
            elif spec.kind == "wood_h":
                return _native.generate_wood_horizontal(
                    spec.cells, pixel_size, wood_r, wood_g, wood_b, wood_a
                )
            elif spec.kind == "wood_v":
                return _native.generate_wood_vertical(
                    spec.cells, pixel_size, wood_r, wood_g, wood_b, wood_a
                )

        # Fallback to Python implementation
        if spec.kind == "tile":
            return MaterialTileRenderer._tile_pixels(spec.cells, pixel_size)

        tile_px = spec.cells * pixel_size
        if spec.kind == "wood_h":
            return MaterialTileRenderer._wood_planks_horizontal(tile_px, pixel_size)
        if spec.kind == "wood_v":
            return MaterialTileRenderer._wood_planks_vertical(tile_px, pixel_size)

        return None

    @staticmethod
    def tile_block(tile: NumpyArray, r0: int, r1: int, c0: int, c1: int) -> NumpyArray:
        """
        Create a tiled block for a specific region using modulo indexing.

        Args:
            tile: The tile pattern to repeat
            r0, r1: Row range
            c0, c1: Column range

        Returns:
            Tiled array for the specified region
        """
        th, tw, _ = tile.shape
        rows = (np.arange(r0, r1) % th).astype(np.intp, copy=False)
        cols = (np.arange(c0, c1) % tw).astype(np.intp, copy=False)
        return tile[rows[:, None], cols[None, :], :]

    @staticmethod
    def apply_overlay_on_region(
        image: NumpyArray,
        tile: NumpyArray,
        r0: int,
        r1: int,
        c0: int,
        c1: int,
    ) -> None:
        """
        Apply a material tile pattern to a region of an image.

        Args:
            image: Target RGBA image to modify (in-place)
            tile: Material tile pattern
            r0, r1: Row range in the image
            c0, c1: Column range in the image
        """
        region = image[r0:r1, c0:c1]
        overlay = MaterialTileRenderer.tile_block(tile, r0, r1, c0, c1)
        mask = overlay[..., 3] > 0
        if np.any(mask):
            region[mask] = overlay[mask]

