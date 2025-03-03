import cv2
import numpy as np

image = cv2.imread("datasets/S2_201804_RGB.png", cv2.IMREAD_UNCHANGED)
h, w, _ = image.shape

tile_size = 1000
for i in range(0, w, tile_size):
    for j in range(0, h, tile_size):
        tile = image[j:j+tile_size, i:i+tile_size]
        cv2.imwrite(f"tile_{i}_{j}.png", tile)
