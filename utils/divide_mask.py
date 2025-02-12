import os
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from colormap import land_cover_cmap 

def create_tiles_from_mask(mask_file, output_dir, num_classes=42):
    os.makedirs(output_dir, exist_ok=True)

    empty_tiles = []
    count = 0
    
    # Definir un colormap personalizado con colores distintos para cada clase
    colors = plt.cm.get_cmap('tab20', num_classes)  # Usa el colormap 'tab20' con 42 colores
    cmap = ListedColormap(colors(range(num_classes)))  # Crea un colormap personalizado

    with rasterio.open(mask_file) as src:
        img_width, img_height = src.width, src.height
        tile_size = 512
        
        # Recorrer toda la imagen creando tiles de 1000x1000
        for x in range(0, img_width, tile_size):
            for y in range(0, img_height, tile_size):

                window = Window(x, y, tile_size, tile_size)

                # Leer la banda de la máscara (única banda)
                try:
                    image = src.read(1, window=window)
                    if image.size == 0:
                        print(f"Skipping empty tile at ({x}, {y})")
                        continue
                except Exception as e:
                    print(f"Error reading tile at ({x}, {y}): {e}")
                    continue

                # Comprobar si la imagen está vacía (todos los valores son cero)
                if np.all(image == 0):
                    empty_tiles.append((x, y))
                    print(f"Empty tile at ({x}, {y})")
                    continue 

                # Asegurarse de que las clases estén dentro del rango esperado
                image[image >= num_classes] = num_classes - 1  # Cortar valores fuera de rango

                image_colored = land_cover_cmap(image) 

                # Guardar la máscara como un archivo PNG con colores
                filepath = os.path.join(output_dir, f"tile_{x}_{y}.png")
                plt.imsave(filepath, image_colored)  # Guarda la máscara con colores
                
                count += 1
                print(f"Saved mask tile {count}: for {x} - {y}")
    
    # Imprimir los resultados de los tiles vacíos
    if empty_tiles:
        print("\nEmpty tiles:")
        for tile in empty_tiles:
            print(f"Tile at coordinates: {tile}")
    else:
        print("\nNo empty tiles found.")
    
    print(f"Total mask tiles saved: {count}")
    print(f"Total empty tiles: {len(empty_tiles)}")

# Llamada a la función para crear los tiles desde la máscara
create_tiles_from_mask('./rescale_mask/mask.tif', './mask', num_classes=42)