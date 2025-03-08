import os
import shutil
import cv2
import numpy as np

# Directorios de entrada
IMG_DIR = "./train/img"
MASK_DIR = "./train/mask"

# Directorios de salida
LOW_INFO_IMG_DIR = "./train/low_info_img"
LOW_INFO_MASK_DIR = "./train/low_info_mask"

# Crear las carpetas de salida si no existen
os.makedirs(LOW_INFO_IMG_DIR, exist_ok=True)
os.makedirs(LOW_INFO_MASK_DIR, exist_ok=True)

# Umbral mínimo de píxeles activos en la máscara
THRESHOLD = 140000

# Obtener listas de archivos asegurando que tengan la misma base de nombre
img_files = {f.split('.')[0]: f for f in os.listdir(IMG_DIR)}
mask_files = {f.split('.')[0]: f for f in os.listdir(MASK_DIR)}

# Filtrar solo archivos que tienen imagen y máscara
common_files = set(img_files.keys()) & set(mask_files.keys())

# Mover archivos con máscaras poco informativas
moved_count = 0
for base_name in common_files:
    img_name = img_files[base_name]
    mask_name = mask_files[base_name]

    img_path = os.path.join(IMG_DIR, img_name)
    mask_path = os.path.join(MASK_DIR, mask_name)

    # Cargar la máscara en escala de grises
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Error cargando máscara: {mask_path}")
        continue

    # Contar píxeles no negros
    nonzero_pixels = np.count_nonzero(mask)

    if nonzero_pixels <= THRESHOLD:
        # Mover imagen y máscara a las carpetas correspondientes
        shutil.move(img_path, os.path.join(LOW_INFO_IMG_DIR, img_name))
        shutil.move(mask_path, os.path.join(LOW_INFO_MASK_DIR, mask_name))
        moved_count += 1
        print(f"Moviendo {img_name} y {mask_name} - Píxeles útiles: {nonzero_pixels}")

print(f"\nProceso completado. {moved_count} imágenes y máscaras movidas.")