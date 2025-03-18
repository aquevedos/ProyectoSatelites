import os
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from colormap import land_cover_cmap

'''
This code splits a mask image into smaller 300x300 tiles and discards the tiles that are completely black (all 0).
It reads the mask file, iterates over it in steps of 300 pixels, checks each tile, and saves it if it contains non-zero values. 
'''

def create_tiles_from_mask(mask_file, output_dir, num_classes=12):
    os.makedirs(output_dir, exist_ok=True)

    empty_tiles = []
    count = 0
    tile_size = 300  # Size of tiles

    with rasterio.open(mask_file) as src:
        img_width, img_height = src.width, src.height

        for x in range(0, img_width, tile_size):
            for y in range(0, img_height, tile_size):
                window = Window(x, y, tile_size, tile_size)

                try:
                    image = src.read(1, window=window)
                    if image.size == 0:
                        print(f"Skipping empty tile at ({x}, {y})")
                        continue
                except Exception as e:
                    print(f"Error reading tile at ({x}, {y}): {e}")
                    continue

                # Check if the tile is empty (all values are zero)
                if np.all(image == 0):
                    empty_tiles.append((x, y))
                    print(f"Empty tile at ({x}, {y})")
                    continue 

                image = np.clip(image, 0, num_classes - 1)

                image_colored = land_cover_cmap(image)

                filepath = os.path.join(output_dir, f"tile_{x}_{y}.png")
                plt.imsave(filepath, image_colored)  
                
                count += 1
                print(f"Saved mask tile {count}: for {x} - {y}")
    
    if empty_tiles:
        print("\nEmpty tiles:")
        for tile in empty_tiles:
            print(f"Tile at coordinates: {tile}")
    else:
        print("\nNo empty tiles found.")
    
    print(f"Total mask tiles saved: {count}")
    print(f"Total empty tiles: {len(empty_tiles)}")

# Mask with ONLY 12 classes
create_tiles_from_mask('./rescale_mask/mask.tif', './train/mask300', num_classes=12)