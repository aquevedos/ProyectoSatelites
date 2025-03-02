"""
Autor: Esteve Graells

Utilitzo aquest script per a comptar tiles de 1000x1000 en un fitxer Geotiff, processant els tiles en memòria. 
Els tiles que son 0 no els vull incloure en l'entrenament del model i per tant els excloc de la mostra.

"""

import rasterio
import numpy as np

# Path to the large GeoTIFF file
geotif_path = "G:\\My Drive\\Personal\\PostgrauAIDL\\Proyecto-Grupo\\datasets\\LC_2018_UTM_WGS84_31N_1m.tif"
geotif_path = "G:\\My Drive\\Personal\\PostgrauAIDL\\Proyecto-Grupo\\datasets\\LC_2018_UTM_WGS84_31N_1m_12_Classes.tif"

# Tile size
tile_size = 1000

# Open the GeoTIFF file
with rasterio.open(geotif_path) as src:
    # Get total image dimensions
    width, height = src.width, src.height

    # List to store empty tile coordinates
    empty_tiles = []

    # Loop through tiles
    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):
            # Llegit tiles de 1000x1000 
            tile = src.read(1, window=rasterio.windows.Window(col, row, tile_size, tile_size))
            
            # Check if the entire tile is zero
            if np.all(tile == 0):
                empty_tiles.append((row, col)) 
                print(f"Tile at ({row}, {col}) is empty")

print(f"Total empty tiles: {len(empty_tiles)}")
#print(empty_tiles)
