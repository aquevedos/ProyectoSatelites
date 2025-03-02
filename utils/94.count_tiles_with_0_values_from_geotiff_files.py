"""
Autor: Esteve Graells

Utilitzo aquest script per a trobar i comptar fitxers GeoTIFF que continguin només valors de zero en tiles.
L'anterior script el que ho fa es validar-ho en un fitxer GEOTIFF sense trocejar.

"""


import os
import rasterio
import numpy as np

def find_blank_geotiffs(folder):
    """
    """
    blank_files = []
    total = 0
    blanks = 0

    for filename in os.listdir(folder):
        if filename.endswith(".tif"):
            file_path = os.path.join(folder, filename)
            
            with rasterio.open(file_path) as src:
                band = src.read(1)
                
                if np.all(band == 0):
                    blank_files.append(filename)
                    blanks += 1
                    src.close()

                    new_filename = filename.replace(".tif", "_blank.tif")
                    new_file_path = os.path.join(folder, new_filename)
                    #os.rename(file_path, new_file_path)
                    


        total += 1
    
    return blank_files, total, blanks

if __name__ == "__main__":
    folder_path = "D:\\tiles_training\\tiles_masks\\tiffs"  # On tinc tots els tiffs
    blank_files, total, blanks  = find_blank_geotiffs(folder_path)
    print(f"Total: {total} - Blank files {blanks}")
    