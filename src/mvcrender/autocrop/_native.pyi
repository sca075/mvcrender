from typing import Tuple
import numpy as np

class AutoCrop:
    """C-Python implementation of AutoCrop (1:1 API with Python)"""
    def __init__(self, handler: object) -> None: ...
    def init_auto_crop(self) -> list[int] | None: ...
    def image_margins(self, image_array: np.ndarray, detect_colour: Tuple[int,int,int,int]) -> tuple[int,int,int,int]: ...
    def check_if_zoom_is_on(self, image_array: np.ndarray, *, margin_size: int = 100, zoom: bool = False, rand256: bool = False) -> np.ndarray: ...
    def rotate_image(self, image_array: np.ndarray, rotate: int) -> np.ndarray: ...
    def auto_trim_and_zoom_image(
        self,
        image_array: np.ndarray,
        detect_colour: Tuple[int,int,int,int] = (93,109,126,255),
        margin_size: int = 0,
        rotate: int = 0,
        zoom: bool = False,
        rand256: bool = False,
    ) -> np.ndarray:
        """ Crops the image based on the background color. """
        ...
