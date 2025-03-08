import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def combine_tiles(img_dir, mask_dir, output_dir):
    # Crear la carpeta de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Listar los archivos de imágenes y máscaras
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    # Si hay menos máscaras que imágenes, ajustamos la cantidad de imágenes a las máscaras
    num_files_to_combine = min(len(img_files), len(mask_files))
    
    print(f"Combinando {num_files_to_combine} imágenes y máscaras.")
    
    # Recorrer los archivos de las imágenes y máscaras y combinarlos
    for i in range(num_files_to_combine):
        img_file = img_files[i]
        mask_file = mask_files[i]

        # Cargar la imagen y la máscara
        img_path = os.path.join(img_dir, img_file)
        mask_path = os.path.join(mask_dir, mask_file)

        img = Image.open(img_path)
        mask = Image.open(mask_path)

        # Convertir la imagen y la máscara a numpy arrays
        img_array = np.array(img)
        mask_array = np.array(mask)

        # Comprobar si la imagen es en escala de grises (1 canal) o en color (3 canales)
        if img_array.ndim == 2:  # Imagen en escala de grises (1 canal)
            img_array = np.expand_dims(img_array, axis=-1)  # Convertir a 3 canales (gris, gris, gris)
        
        if mask_array.ndim == 2:  # Máscara en escala de grises (1 canal)
            mask_array = np.expand_dims(mask_array, axis=-1)  # Convertir a 3 canales (binario, binario, binario)

        # Combinamos la imagen y la máscara como 4 canales: RGB + Alpha (máscara como canal alfa)
        combined_array = np.concatenate((img_array, mask_array), axis=-1)  # (Imagen RGB + Máscara como canal alfa)

        # Convertir el array combinado a una imagen
        combined_img = Image.fromarray(combined_array.astype(np.uint8))

        # Guardar la imagen combinada
        combined_img_path = os.path.join(output_dir, f"combined_{img_file}")
        combined_img.save(combined_img_path)

        print(f"Combinado y guardado {combined_img_path}")

    print("Combinación completa.")

# Directorios donde están los tiles de imagen y máscara
img_dir = './img'  # Carpeta con los tiles de las imágenes
mask_dir = './mask'  # Carpeta con los tiles de las máscaras
output_dir = './combined_output'  # Directorio donde se guardarán las imágenes combinadas

# Llamar a la función para combinar los tiles
combine_tiles(img_dir, mask_dir, output_dir)