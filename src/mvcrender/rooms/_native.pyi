"""Type stubs for mvcrender.rooms._native"""

import numpy as np
from numpy.typing import NDArray

def process_room_mask(
    mask: NDArray[np.uint8],
    padding: int = 10
) -> NDArray[np.uint8]:
    """
    Apply morphological operations (erosion + dilation) to clean room mask.
    
    Args:
        mask: 2D uint8 numpy array (binary mask)
        padding: Padding size (default: 10)
    
    Returns:
        Processed 2D uint8 numpy array
    """
    ...

def fill_compressed_pixels_mask(
    pixel_data: NDArray[np.int32],
    width: int,
    height: int,
    min_x: int,
    min_y: int
) -> NDArray[np.uint8]:
    """
    Create mask from compressed pixel format [x, y, length, ...].
    
    Args:
        pixel_data: Nx3 int32 numpy array of (x, y, length) triplets
        width: Width of output mask
        height: Height of output mask
        min_x: X offset to subtract from coordinates
        min_y: Y offset to subtract from coordinates
    
    Returns:
        2D uint8 numpy array (binary mask)
    """
    ...

def extract_points_from_mask(
    mask: NDArray[np.uint8]
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """
    Extract all non-zero points from binary mask.
    
    Args:
        mask: 2D uint8 numpy array (binary mask)
    
    Returns:
        Tuple of (x_coords, y_coords) as int32 numpy arrays
    """
    ...

