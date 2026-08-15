"""Generate a handful of synthetic 'microscopy-like' images to test the QC pipeline:
sharp+normal, blurry, too dark, too bright, and empty (no cells)."""
import numpy as np
import cv2
import os

np.random.seed(0)
out_dir = "test_images"
os.makedirs(out_dir, exist_ok=True)


def make_cells_image(n_cells=200, size=512, base_brightness=60, blur_ksize=0):
    img = np.full((size, size), base_brightness, dtype=np.float64)
    for _ in range(n_cells):
        cx, cy = np.random.randint(10, size - 10, size=2)
        r = np.random.randint(4, 9)
        intensity = base_brightness + np.random.randint(120, 180)
        cv2.circle(img, (cx, cy), r, intensity, -1)
    img += np.random.normal(0, 4, size=img.shape)  # sensor noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    if blur_ksize > 0:
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    return img


# 1. Normal, sharp, well-exposed
cv2.imwrite(f"{out_dir}/img001_normal.png", make_cells_image(n_cells=220, base_brightness=60))

# 2. Blurry (heavy gaussian blur)
cv2.imwrite(f"{out_dir}/img002_blurry.png", make_cells_image(n_cells=220, base_brightness=60, blur_ksize=15))

# 3. Too dark
cv2.imwrite(f"{out_dir}/img003_dark.png", make_cells_image(n_cells=200, base_brightness=5))

# 4. Too bright / saturated
cv2.imwrite(f"{out_dir}/img004_bright.png", make_cells_image(n_cells=200, base_brightness=245))

# 5. Empty field (no cells)
empty = np.full((512, 512), 60, dtype=np.uint8)
empty = empty + np.random.normal(0, 3, size=empty.shape).astype(np.uint8)
cv2.imwrite(f"{out_dir}/img005_empty.png", empty)

# 6. A 16-bit TIFF version of the normal image (simulate real scientific camera output)
import tifffile
img16 = (make_cells_image(n_cells=180, base_brightness=60).astype(np.uint16)) * 200
tifffile.imwrite(f"{out_dir}/img006_16bit.tif", img16)

print("Test images written to", out_dir)
for f in sorted(os.listdir(out_dir)):
    print(" -", f)
