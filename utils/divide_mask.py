import os
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from colormap import land_cover_cmap

def create_tiles_from_mask(mask_file, output_dir, num_classes=12):
    os.makedirs(output_dir, exist_ok=True)

    empty_tiles = []
    count = 0
    tile_size = 300  # Tamaño de los tiles

    with rasterio.open(mask_file) as src:
        img_width, img_height = src.width, src.height

        # Recorrer toda la imagen creando tiles de 512x512
        for x in range(0, img_width, tile_size):
            for y in range(0, img_height, tile_size):
                window = Window(x, y, tile_size, tile_size)

                try:
                    image = src.read(1, window=window)  # Leer la banda de la máscara
                    if image.size == 0:
                        print(f"Skipping empty tile at ({x}, {y})")
                        continue
                except Exception as e:
                    print(f"Error reading tile at ({x}, {y}): {e}")
                    continue

                # Comprobar si el tile está vacío (todos los valores son cero)
                if np.all(image == 0):
                    empty_tiles.append((x, y))
                    print(f"Empty tile at ({x}, {y})")
                    continue 

                # Asegurarse de que los valores estén dentro del rango esperado [0, num_classes - 1]
                image = np.clip(image, 0, num_classes - 1)

                # Aplicar el colormap **solo con los primeros 12 colores**
                image_colored = land_cover_cmap(image)  # Usamos la paleta de colores importada

                # Guardar la imagen como un archivo PNG con colores
                filepath = os.path.join(output_dir, f"tile_{x}_{y}.png")
                plt.imsave(filepath, image_colored)  
                
                count += 1
                print(f"Saved mask tile {count}: for {x} - {y}")
    
    # Imprimir los tiles vacíos encontrados
    if empty_tiles:
        print("\nEmpty tiles:")
        for tile in empty_tiles:
            print(f"Tile at coordinates: {tile}")
    else:
        print("\nNo empty tiles found.")
    
    print(f"Total mask tiles saved: {count}")
    print(f"Total empty tiles: {len(empty_tiles)}")

# Llamada a la función para crear los tiles desde la máscara con SOLO 12 clases
create_tiles_from_mask('./rescale_mask/mask.tif', './train/mask300', num_classes=12)