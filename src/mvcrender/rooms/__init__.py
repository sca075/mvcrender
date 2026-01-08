"""
Rooms processing module for mvcrender.

Provides high-performance C implementations of room boundary detection
and processing for vacuum map data.
"""

from mvcrender.rooms._native import (
    process_room_mask,
    fill_compressed_pixels_mask,
    extract_points_from_mask,
)

__all__ = [
    "process_room_mask",
    "fill_compressed_pixels_mask",
    "extract_points_from_mask",
]

