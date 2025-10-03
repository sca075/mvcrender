import numpy as np
from mvcrender.autocrop import AutoCrop

import time
from PIL import Image

class DummyShared:
    def __init__(self):
        self.trims = type("T", (), {"to_dict": lambda self: {"trim_up":0,"trim_down":0,"trim_left":0,"trim_right":0}})()
        self.offset_top=0; self.offset_down=0; self.offset_left=0; self.offset_right=0
        self.vacuum_state="cleaning"; self.image_auto_zoom=True
        self.image_ref_width=0; self.image_ref_height=0

class DummyBaseHandler:
    def __init__(self):
        self.crop_img_size = [0, 0]
        self.crop_area = None  # [0,0,0,0]DummyShared()
        self.shared = None
        self.file_name = "smoke"
        self.robot_position = (200, 150)
        self.robot_pos = {"in_room": None}

class DummyHandler(DummyBaseHandler, AutoCrop):
    def __init__(self, shared=None):
        DummyBaseHandler.__init__(self)
        self.shared = shared
        AutoCrop.__init__(self, self)
        self.max_frames = 0
        self.room_propriety = None
        self.rooms_pos = []
        self.img_size = (0, 0)

# Init simple handler as done in Valetudo Map Parsers Library
h = DummyHandler(DummyShared())

H,W = 5700, 5700
img = np.zeros((H,W,4), dtype=np.uint8)
img[...,3] = 255
img[:, :, :3] = (93,109,126)  # bg
img[500:2500, 800:3200, :3] = (120,200,255)  # fg block

# Averaged timing for stability
# The first run is slower as the CropArea is computed
# Subsequent runs are faster as the CropArea is cached
# Compare with Python implementation we gain a factor of 2-3x speedup

runs_avg = 2
start = time.perf_counter()
res_avg_img: np.ndarray = img # just to make mypy happy
for _ in range(runs_avg):
    start_single = time.perf_counter()
    res_avg_img = h.auto_trim_and_zoom_image(
        img, (93,109,126,255),
        margin_size=10,
        rotate=90,
        zoom=False,
        rand256=True,
    )
    single_total_ms = (time.perf_counter() - start_single) * 1000.0
    print(f"Run {_} - total time: {single_total_ms:.3f} ms")

elapsed_total_ms = (time.perf_counter() - start) * 1000.0  / runs_avg

# Show one resulting image so visual check remains possible
res_img = Image.fromarray(res_avg_img)
print(f"Resulting image: {res_img.size}")
res_img.show()

# Report timing and output image size and confirm cropping worked.
print(f"out shape: {res_avg_img.shape} crop_img_size: {h.crop_img_size} crop_area: {h.crop_area}")
print(f"avg total: {elapsed_total_ms:.3f} ms over {runs_avg} runs")
print(f"shared data: {h.shared.image_ref_width}x{h.shared.image_ref_height}")
print(f"hander data: {h.img_size}")
