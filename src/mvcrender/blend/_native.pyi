from typing import Tuple
import numpy as np

def get_blended_color(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    arr: np.ndarray,          # HxWx4, uint8
    color: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    """
    Segment-aware blended color (5px offset endpoints averaged).
    """
    ...

def blend_mask_inplace(
    image_array: np.ndarray,  # HxWx4, uint8
    mask: np.ndarray,         # HxW, bool
    foreground_color: Tuple[int, int, int, int],
) -> None:
    """
    Blend RGBA into image wherever mask==True (straight alpha).
    """
    ...

def sample_and_blend_color(
    arr: np.ndarray,
    x: int,
    y: int,
    color: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    """
    Return fg OVER image[x,y] using straight alpha.
    """
    ...
