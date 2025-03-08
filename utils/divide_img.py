import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from rasterio.windows import Window
from find_missing_files import find_missing_files

def create_tiles_from_img(sentinel_file, mask_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(sentinel_file) as src:
        img_width, img_height = src.width, src.height
        tile_size = 300
        count = 0
        
        for x in range(0, img_width, tile_size):
            for y in range(0, img_height, tile_size):
                tile_name = f"tile_{x}_{y}.png"
                mask_path = os.path.join(mask_dir, tile_name)
                
                if not os.path.exists(mask_path):
                    print(f"Skipping tile at ({x}, {y}) due to missing mask")
                    continue
                
                window = Window(x, y, tile_size, tile_size)
                
                try:
                    image = src.read([1, 2, 3], window=window)
                    with rasterio.open(mask_path) as mask_src:
                        mask = mask_src.read(1)
                    
                    if image.size == 0 or mask.size == 0:
                        print(f"Skipping empty tile at ({x}, {y})")
                        continue
                    
                    # Reordenar a RGB sin modificar los valores
                    image = image[[2, 1, 0]]
                    
                    # Convertir los valores de la imagen de uint16 a uint8
                    image = image.astype(np.float32)
                    image = (image / image.max()) * 255
                    image = np.clip(image, 0, 255).astype(np.uint8)
                    
                    # Verificar si hay al menos un píxel válido en la máscara
                    if np.any(mask > 0):
                        image_rescaled = np.moveaxis(image, 0, -1)
                        filepath = os.path.join(output_dir, f"tile_{x}_{y}.png")
                        plt.imsave(filepath, image_rescaled)
                        count += 1
                        print(f"Saved image {count}: for {x} - {y}")
                    else:
                        print(f"Skipping tile at ({x}, {y}) due to mask")
                except Exception as e:
                    print(f"Error processing tile at ({x}, {y}): {e}")
                    continue
    
    print(f"Total images saved: {count}")


if __name__ == "__main__":
    mask_dir = './train/mask300'
    output_dir = './train/img300'
    create_tiles_from_img('./datasets/img.tif', mask_dir, output_dir)

    # Buscar las diferencias para eliminar aqeullas imágenes que no tienen máscara
    find_missing_files(mask_dir, output_dir)

