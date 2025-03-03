import re
import os
from PIL import Image

def get_tile_coordinates(filename, subname):
    match = re.match(f'{subname}(\d+)_(\d+)\.png', filename) # 321 perquè és la combinació de 3, 2 1 de Sentinel q he extret si és diferent, cal canviar
    if match:
        return int(match.group(1)), int(match.group(2))
    return None

def merge_tiles(tile_dir, output_file, subname):
    tiles = []
    
    # Scan directory for tile images
    for filename in os.listdir(tile_dir):
        coords = get_tile_coordinates(filename, subname)
        if coords:
            tiles.append((coords, filename))
    
    if not tiles:
        print("No tiles found!")
        return
    
    # Determine grid size
    max_x = max(tile[0][0] for tile in tiles)
    max_y = max(tile[0][1] for tile in tiles)
    
    # Compute final image size
    width = (max_x // 1000 + 1) * 1000
    height = (max_y // 1000 + 1) * 1000
    
    print(f"Merging {len(tiles)} tiles into {width}x{height} image...")
    
    # Create blank canvas
    merged_image = Image.new('RGB', (width, height))
    
    # Place tiles on the canvas with a counter
    counter = 0
    for (x, y), filename in tiles:
        tile_path = os.path.join(tile_dir, filename)
        tile_image = Image.open(tile_path)
        merged_image.paste(tile_image, (x, y))
        counter += 1
        print(f"Processed {counter}/{len(tiles)} tiles")
    
    # Save merged image with optimized speed
    merged_image.save(output_file, compress_level=0)
    print(f"Merged image saved as {output_file}")

merge_tiles("D:\\tiles_training\\tiles_masks", "D:\\tiles_training\\tiles_masks\\12classes_downsampled.png", "mask_tile_") # Tarda uns 90'' 