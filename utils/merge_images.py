import os
import numpy as np
from PIL import Image

def combine_tiles(input_dir, output_file, tile_size=(512, 512), grid_size=(30, 30)):
    """
    Combina tiles en una sola imagen grande.

    Args:
        input_dir (str): Directorio donde se encuentran las imágenes recortadas.
        output_file (str): Ruta del archivo donde se guardará la imagen combinada.
        tile_size (tuple): Tamaño de cada tile (ancho, alto).
        grid_size (tuple): Número de tiles en la cuadrícula (columnas, filas).
    """
    # Crear una matriz vacía para almacenar la imagen combinada (con 3 canales para los colores)
    combined_image = np.zeros((grid_size[1] * tile_size[1], grid_size[0] * tile_size[0], 3), dtype=np.uint8)

    # Recorrer todas las imágenes en el directorio
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            # Extraer las coordenadas del archivo (asume formato "tile_x_y.png")
            try:
                _, x_str, y_str = filename[:-4].split('_')
                x, y = int(x_str), int(y_str)
            except ValueError:
                print(f"Skipping file with unexpected name format: {filename}")
                continue

            # Verificar que las coordenadas estén dentro del rango de la cuadrícula
            col, row = x % grid_size[0], y % grid_size[1]  # Calculamos las posiciones correctas dentro de la cuadrícula
            x_offset, y_offset = col * tile_size[0], row * tile_size[1]

            # Asegurarnos de que la posición no se salga del límite de la imagen combinada
            if x_offset + tile_size[0] > combined_image.shape[1] or y_offset + tile_size[1] > combined_image.shape[0]:
                print(f"Skipping tile {filename}: position out of bounds")
                continue

            # Cargar el tile y convertirlo a un array de Numpy
            tile_path = os.path.join(input_dir, filename)
            tile = Image.open(tile_path)
            tile = np.array(tile)

            # Verificar si la imagen tiene 4 canales (RGB + Alfa), si es así, tomar solo los primeros 3 canales (RGB)
            if tile.shape[2] == 4:
                tile = tile[:, :, :3]  # Extraer solo los 3 canales de color (RGB)

            # Verificar si las dimensiones del tile coinciden con el tamaño esperado
            if tile.shape != (tile_size[1], tile_size[0], 3):  # 3 canales (RGB)
                print(f"Skipping tile {filename}: unexpected size {tile.shape}")
                continue

            # Colocar el tile en la posición correspondiente de la imagen combinada
            combined_image[y_offset:y_offset + tile_size[1], x_offset:x_offset + tile_size[0]] = tile

    # Guardar la imagen combinada
    combined_image_pil = Image.fromarray(combined_image)
    combined_image_pil.save(output_file)
    print(f"Imagen combinada guardada en {output_file}")

# Parámetros
input_dir = "./img"  # Carpeta donde están los tiles
output_file = "combined_img.png"  # Archivo de salida
tile_size = (512, 512)  # Tamaño de cada tile
grid_size = (30, 30)  # Dimensiones de la cuadrícula de tiles (columnas, filas)

# Ejecutar la función
combine_tiles(input_dir, output_file, tile_size, grid_size)