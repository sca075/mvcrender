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

class DummyHandler(DummyBaseHandler, AutoCrop):
    def __init__(self, shared=None):
        DummyBaseHandler.__init__(self)
        self.shared = shared
        AutoCrop.__init__(self, self)

        self.file_name = "smoke"
        self.max_frames = 0
        self.robot_pos = {"in_room": None}
        self.robot_position = (200, 150)
        self.crop_img_size = [0,0]
        self.crop_area = None # [0,0,0,0]
        self.room_propriety = None
        self.rooms_pos = []

h = DummyHandler(DummyShared())
# ac = AutoCrop(handler=h)

H,W = 5700, 5700
img = np.zeros((H,W,4), dtype=np.uint8)
img[...,3] = 255
img[:, :, :3] = (93,109,126)  # bg
img[500:2500, 800:3200, :3] = (120,200,255)  # fg block

# Warm-up to avoid first-call overhead

# Averaged timing for stability
runs_avg = 100
start = time.perf_counter()
for _ in range(runs_avg):
    start_single = time.perf_counter()
    res_avg_img = h.async_auto_trim_and_zoom_image(
        img, (93,109,126,255),
        margin_size=10,
        rotate=270,
        zoom=False,
        rand256=True,
    )
    single_total_ms = (time.perf_counter() - start_single) * 1000.0
    print(f"single total: {single_total_ms:.3f} ms")

elapsed_total_ms = (time.perf_counter() - start) * 1000.0  / runs_avg

# Show one resulting image so visual check remains possible
res_img = Image.fromarray(res_avg_img)
res_img.show()

print(f"out shape: {res_avg_img.shape} crop_img_size: {h.crop_img_size} crop_area: {h.crop_area}")
print(f"avg total: {elapsed_total_ms:.3f} ms over {runs_avg} runs")
