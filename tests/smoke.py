import numpy as np
from mvcrender.autocrop import AutoCrop
from mvcrender.blend import get_blended_color, sample_and_blend_color, blend_mask_inplace
from mvcrender.draw import line_u8, polyline_u8, circle_u8, polygon_u8

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
        self.robot_position = (200, 150, 0)
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

print("=" * 60)
print("mvcrender Smoke Test - Drawing & Blending Functions")
print("=" * 60)

# Init simple handler as done in Valetudo Map Parsers Library
h = DummyHandler(DummyShared())

H, W = 5700, 5700
img = np.zeros((H, W, 4), dtype=np.uint8)
img[..., 3] = 255
img[:, :, :3] = (93, 109, 126)  # bg
img[500:2500, 800:3200, :3] = (120, 200, 255)  # fg block

print("\n1. Testing Drawing Functions")
print("-" * 60)

# 1a. Line with thickness
print("  - Drawing line (red, thickness=5)")
t0 = time.perf_counter()
line_u8(img, 1000, 1000, 2000, 1500, (255, 0, 0, 255), 5)
print(f"    Time: {(time.perf_counter() - t0) * 1000:.3f} ms")

# 1b. Polyline
print("  - Drawing polyline (yellow, thickness=3)")
xs = np.array([1200, 1400, 1600, 1800, 2000], dtype=np.int32)
ys = np.array([1700, 1600, 1800, 1650, 1750], dtype=np.int32)
t0 = time.perf_counter()
polyline_u8(img, xs, ys, (255, 255, 0, 255), 3)
print(f"    Time: {(time.perf_counter() - t0) * 1000:.3f} ms")

# 1c. Circle filled
print("  - Drawing filled circle (cyan, radius=80)")
t0 = time.perf_counter()
circle_u8(img, 1500, 2000, 80, (0, 255, 255, 255), -1)
print(f"    Time: {(time.perf_counter() - t0) * 1000:.3f} ms")

# 1d. Circle outlined
print("  - Drawing outlined circle (magenta, radius=100, thickness=4)")
t0 = time.perf_counter()
circle_u8(img, 2200, 2000, 100, (255, 0, 255, 255), 4)
print(f"    Time: {(time.perf_counter() - t0) * 1000:.3f} ms")

# 1e. Polygon with fill
print("  - Drawing polygon (orange outline, light blue fill)")
poly_xs = np.array([2500, 2700, 2800, 2600, 2400], dtype=np.int32)
poly_ys = np.array([1200, 1300, 1500, 1600, 1400], dtype=np.int32)
t0 = time.perf_counter()
polygon_u8(img, poly_xs, poly_ys, (255, 128, 0, 255), 3, (173, 216, 230, 255))
print(f"    Time: {(time.perf_counter() - t0) * 1000:.3f} ms")

print("\n2. Testing Blending Functions")
print("-" * 60)

# 2a. sample_and_blend_color - blend a small patch
print("  - sample_and_blend_color: semi-transparent green patch (200x200)")
green_overlay = (50, 220, 50, 128)
t0 = time.perf_counter()
for y in range(1400, 1600):
    for x in range(1900, 2100):
        r, g, b, a = sample_and_blend_color(img, x, y, green_overlay)
        img[y, x] = [r, g, b, a]
print(f"    Time: {(time.perf_counter() - t0) * 1000:.3f} ms")

# 2b. get_blended_color - segment-aware blending
print("  - get_blended_color: semi-transparent red line (segment-aware)")
red_overlay = (220, 50, 50, 128)
t0 = time.perf_counter()
x0, y0, x1, y1 = 2800, 1800, 3000, 2200
for t in np.linspace(0, 1, 100):
    x = int(x0 + t * (x1 - x0))
    y = int(y0 + t * (y1 - y0))
    r, g, b, a = get_blended_color(x0, y0, x1, y1, img, red_overlay)
    if 0 <= y < H and 0 <= x < W:
        img[y, x] = [r, g, b, a]
print(f"    Time: {(time.perf_counter() - t0) * 1000:.3f} ms")

# 2c. blend_mask_inplace - mask-based blending
print("  - blend_mask_inplace: semi-transparent blue via circular mask")
mask = np.zeros((H, W), dtype=bool)
cy, cx, radius = 2200, 2800, 120
yy, xx = np.ogrid[:H, :W]
mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
blue_overlay = (50, 100, 255, 150)
t0 = time.perf_counter()
blend_mask_inplace(img, mask, blue_overlay)
print(f"    Time: {(time.perf_counter() - t0) * 1000:.3f} ms")

print("\n3. Testing AutoCrop with Rotation")
print("-" * 60)

runs_avg = 2
start = time.perf_counter()
res_avg_img: np.ndarray = img
for run in range(runs_avg):
    start_single = time.perf_counter()
    res_avg_img = h.auto_trim_and_zoom_image(
        img, (93, 109, 126, 255),
        margin_size=10,
        rotate=90,
        zoom=False,
        rand256=True,
    )
    single_total_ms = (time.perf_counter() - start_single) * 1000.0
    print(f"  Run {run + 1} - total time: {single_total_ms:.3f} ms")

elapsed_total_ms = (time.perf_counter() - start) * 1000.0 / runs_avg

print("\n" + "=" * 60)
print("Results Summary")
print("=" * 60)
print(f"Output shape: {res_avg_img.shape}")
print(f"Crop img size: {h.crop_img_size}")
print(f"Crop area: {h.crop_area}")
print(f"Avg autocrop time: {elapsed_total_ms:.3f} ms over {runs_avg} runs")
print(f"Shared data: {h.shared.image_ref_width}x{h.shared.image_ref_height}")
print(f"Handler data: {h.img_size}")

# Show resulting image
res_img = Image.fromarray(res_avg_img)
print(f"\nDisplaying resulting image: {res_img.size}")
res_img.show()

print("\n✅ All drawing and blending functions tested successfully!")
