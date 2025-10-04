from typing import Tuple, Optional
import numpy as np

Color = Tuple[int, int, int] | Tuple[int, int, int, int]

def line_u8(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: Color, thickness: int) -> tuple[int,int,int,int]:
    """
    Draw a solid line with thickness (overwrite RGBA).
    image: HxWx4, dtype=uint8, C-contiguous
    x0, y0, x1, y1: endpoints of line
    color: (r,g,b) or (r,g,b,a)
    thickness: int (pixels)
    Returns: (x0, y0, x1, y1) ROI touched
    """
    ...
def polyline_u8(image: np.ndarray, xs: np.ndarray, ys: np.ndarray, color: Color, thickness: int) -> tuple[int,int,int,int]:
    """
    Draw a solid polyline with thickness (overwrite RGBA).
    image: HxWx4, dtype=uint8, C-contiguous
    xs, ys: int32 arrays of same length >=2
    color: (r,g,b) or (r,g,b,a)
    thickness: int (pixels)
    Returns: (x0, y0, x1, y1) ROI touched
    """
    ...
def circle_u8(image: np.ndarray, cx: int, cy: int, radius: int, color: Color, thickness: int = -1) -> tuple[int,int,int,int]:
    """
    Draw a circle. thickness<0 => filled; else outlined.
    image: HxWx4, dtype=uint8, C-contiguous
    cx, cy: center of circle
    radius: int (pixels)
    color: (r,g,b) or (r,g,b,a)
    Returns: (x0, y0, x1, y1) ROI touched
    """
    ...
def polygon_u8(image: np.ndarray, xs: np.ndarray, ys: np.ndarray, outline_color: Color, thickness: int, fill_color: Optional[Color] = None) -> tuple[int,int,int,int]:
    """
    Draw polygon outline (thickness) and optional fill.
    image: HxWx4, dtype=uint8, C-contiguous
    xs, ys: int32 arrays of same length >=3
    Returns: (x0, y0, x1, y1) ROI touched
    """
    ...
