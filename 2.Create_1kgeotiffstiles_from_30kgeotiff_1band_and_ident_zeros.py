"""
Autor: Esteve Graells

Utilitzo aquest script per a reduir un GeoTIFF de 30.000 x 30.000 píxels en tiles de 1.000 x 1.000 píxels.
Cada tile es guarda amb la georeferenciació adequada i sense compressió. Si un tile conté
només valors zero, es guarda amb el sufix "_blanks" al nom del fitxer.

Paràmetres:
    input_tif (str): Ruta al GeoTIFF de 30.000 x 30.000 píxels.
    output_dir (str): Directori on es guardaran les tessel·les individuals.
    tile_size (int): Mida de cada tessel·la en píxels (per defecte=1000).
"""

import os
import rasterio
from rasterio.windows import Window

def create_masks(
    input_tif,
    output_dir,
    tile_size=1000
):
    """
    :param input_tif: Path to the 30,000 x 30,000 GeoTIFF.
    :param output_dir: Directory where individual tiles will be saved.
    :param tile_width: Width of each tile in pixels (default=1000).
    :param tile_height: Height of each tile in pixels (default=1000).
    """
    with rasterio.open(input_tif) as src:
        # Get image dimensions
        width = src.width
        height = src.height

        # Calculate the number of tiles
        num_tiles_x = width // tile_size
        num_tiles_y = height // tile_size

        # Iterate over each tile
        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                # Define the window for the current tile
                # Aqui podria haver-hi un problema amb la i i la j
                window = Window(i * tile_size, j * tile_size, tile_size, tile_size)
                print(f"Processing tile {i * tile_size} x {j * tile_size}...")

                # Read the data for the current window
                tile_data = src.read(window=window)
                tile_profile = src.profile.copy()

                # Update the profile for the tile
                tile_profile.update({
                    'height': tile_size,
                    'width': tile_size,
                    'transform': rasterio.windows.transform(window, src.transform)
                })

                # Si els valors de la mask són tot zeros, la guardo amb sufix "blanks" al nom
                if not tile_data.any():
                    output_path = f"{output_dir}/mask_{i * tile_size}_{j * tile_size}_blanks.tif"
                else:
                    output_path = f"{output_dir}/mask_{i * tile_size}_{j * tile_size}.tif"
                

                # Write the tile to a new file
                with rasterio.open(output_path, 'w', **tile_profile) as dst:
                    dst.write(tile_data)

if __name__ == '__main__':
    input_file = r"d:\aidl_projecte\datasets\LC_2018_UTM_WGS84_31N_1m_12_Classes_downsampled30x30_mode.tif"
    output_directory = r"d:\aidl_projecte\tiles_masks\tiffsv2"
    
    create_masks(input_tif=input_file, output_dir=output_directory)

