import os
import numpy as np
from PIL import Image

'''

This script merges individual image tiles into a single combined image. 
It reads tiles from an input directory, determines their position based on filename coordinates, 
and arranges them into a grid of a specified size.

'''

def combine_tiles(input_dir, output_file, tile_size=(512, 512), grid_size=(30, 30)):

    combined_image = np.zeros((grid_size[1] * tile_size[1], grid_size[0] * tile_size[0], 3), dtype=np.uint8)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            try:
                _, x_str, y_str = filename[:-4].split('_')
                x, y = int(x_str), int(y_str)
            except ValueError:
                print(f"Skipping file with unexpected name format: {filename}")
                continue

            col, row = x % grid_size[0], y % grid_size[1]
            x_offset, y_offset = col * tile_size[0], row * tile_size[1]

            if x_offset + tile_size[0] > combined_image.shape[1] or y_offset + tile_size[1] > combined_image.shape[0]:
                print(f"Skipping tile {filename}: position out of bounds")
                continue

            tile_path = os.path.join(input_dir, filename)
            tile = Image.open(tile_path)
            tile = np.array(tile)

            if tile.shape[2] == 4:
                tile = tile[:, :, :3] 

            if tile.shape != (tile_size[1], tile_size[0], 3):
                print(f"Skipping tile {filename}: unexpected size {tile.shape}")
                continue

            combined_image[y_offset:y_offset + tile_size[1], x_offset:x_offset + tile_size[0]] = tile

    combined_image_pil = Image.fromarray(combined_image)
    combined_image_pil.save(output_file)
    print(f"Combined image saved as {output_file}")

input_dir = "./img" 
output_file = "combined_img.png"
tile_size = (512, 512) 
grid_size = (30, 30)

combine_tiles(input_dir, output_file, tile_size, grid_size)