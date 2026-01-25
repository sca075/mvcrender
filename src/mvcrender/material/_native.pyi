"""Type stubs for mvcrender.material._native"""

import numpy as np
from numpy.typing import NDArray

def generate_tile_pattern(
    cells: int,
    pixel_size: int,
    r: int = 40,
    g: int = 40,
    b: int = 40,
    a: int = 45
) -> NDArray[np.uint8]:
    """
    Generate a tile pattern.
    
    Args:
        cells: Number of cells in the pattern
        pixel_size: Size of each pixel/cell
        r, g, b, a: RGBA color values
    
    Returns:
        RGBA numpy array with tile pattern
    """
    ...

def generate_wood_horizontal(
    cells: int,
    pixel_size: int,
    r: int = 40,
    g: int = 40,
    b: int = 40,
    a: int = 38
) -> NDArray[np.uint8]:
    """
    Generate horizontal wood plank pattern.
    
    Args:
        cells: Number of cells in the pattern
        pixel_size: Size of each pixel/cell
        r, g, b, a: RGBA color values
    
    Returns:
        RGBA numpy array with wood pattern
    """
    ...

def generate_wood_vertical(
    cells: int,
    pixel_size: int,
    r: int = 40,
    g: int = 40,
    b: int = 40,
    a: int = 38
) -> NDArray[np.uint8]:
    """
    Generate vertical wood plank pattern.
    
    Args:
        cells: Number of cells in the pattern
        pixel_size: Size of each pixel/cell
        r, g, b, a: RGBA color values
    
    Returns:
        RGBA numpy array with wood pattern
    """
    ...

