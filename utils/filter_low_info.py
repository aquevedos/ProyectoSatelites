import os
import shutil
import cv2
import numpy as np

'''
This script filters and moves images whose corresponding masks contain 140,000 or fewer non-black pixels. 
It ensures that low-information images and masks are separated from the main dataset by relocating them to designated folders.
'''

IMG_DIR = "./train/img"
MASK_DIR = "./train/mask"

LOW_INFO_IMG_DIR = "./train/low_info_img"
LOW_INFO_MASK_DIR = "./train/low_info_mask"

os.makedirs(LOW_INFO_IMG_DIR, exist_ok=True)
os.makedirs(LOW_INFO_MASK_DIR, exist_ok=True)

# Minimum threshold of active pixels in the mask
THRESHOLD = 140000

img_files = {f.split('.')[0]: f for f in os.listdir(IMG_DIR)}
mask_files = {f.split('.')[0]: f for f in os.listdir(MASK_DIR)}

common_files = set(img_files.keys()) & set(mask_files.keys())

moved_count = 0
for base_name in common_files:
    img_name = img_files[base_name]
    mask_name = mask_files[base_name]

    img_path = os.path.join(IMG_DIR, img_name)
    mask_path = os.path.join(MASK_DIR, mask_name)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Error loading mask: {mask_path}")
        continue

    # Counting non-black pixels
    nonzero_pixels = np.count_nonzero(mask)

    if nonzero_pixels <= THRESHOLD:
        shutil.move(img_path, os.path.join(LOW_INFO_IMG_DIR, img_name))
        shutil.move(mask_path, os.path.join(LOW_INFO_MASK_DIR, mask_name))
        moved_count += 1
        print(f"Moving {img_name} and {mask_name} - useful pixels: {nonzero_pixels}")

print(f"\nProcess completed. {moved_count} images and masks moved.")