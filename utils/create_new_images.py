import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from utils.colormap import land_cover_cmap

# Definir rutas
img_dir = "./train/img"
mask_dir = "./train/mask"
new_img_dir = "./new_images"
new_mask_dir = "./new_masks"

# Crear carpetas si no existen
os.makedirs(new_img_dir, exist_ok=True)
os.makedirs(new_mask_dir, exist_ok=True)

# Clases con baja representación
low_rep_classes = {5, 7, 8, 10, 11}
threshold = 0.05 

# Transformaciones de data augmentation
augmentations = {
    "rot90": lambda x: x.rotate(90),
    "rot180": lambda x: x.rotate(180),
    "rot270": lambda x: x.rotate(270),
    "flip_h": lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
    "flip_v": lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
}

# Convertir colormap a array para comparación
color_palette = np.array([np.round(np.array(land_cover_cmap(i)[:3]) * 255).astype(int) for i in range(len(land_cover_cmap.colors))])

# Función para convertir una máscara RGB a índices de clase
def rgb_to_class(mask_rgb):
    mask_array = np.array(mask_rgb)  # Convertimos la máscara a un array numpy (H, W, 3)
    class_mask = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.uint8)

    for class_idx, color in enumerate(color_palette):
        matches = np.all(mask_array == color, axis=-1)  # Comparación exacta de píxeles
        class_mask[matches] = class_idx  # Asignar el índice de la clase correspondiente

    return class_mask

# Función para analizar la distribución de clases en una máscara
def analyze_mask(mask_array):
    total_pixels = mask_array.size
    class_counts = {cls: np.sum(mask_array == cls) / total_pixels for cls in np.unique(mask_array)}
    return class_counts

# Recorrer todas las máscaras
for mask_name in tqdm(os.listdir(mask_dir), desc="Procesando máscaras"):
    mask_path = os.path.join(mask_dir, mask_name)
    img_path = os.path.join(img_dir, mask_name)  # La imagen tiene el mismo nombre que la máscara
    
    # Verificar si la imagen existe
    if not os.path.exists(img_path):
        continue
    
    # Cargar la máscara y convertirla a índices de clase
    mask_rgb = Image.open(mask_path).convert("RGB")
    mask_array = rgb_to_class(mask_rgb)
    class_distribution = analyze_mask(mask_array)

    # Verificar si alguna de las clases minoritarias tiene suficiente representación
    if any(class_distribution.get(cls, 0) > threshold for cls in low_rep_classes):
        image = Image.open(img_path).convert("RGB")  # Cargar la imagen asociada
        
        # Aplicar cada transformación y guardar
        for aug_name, aug_fn in augmentations.items():
            new_mask = aug_fn(mask_rgb)
            new_img = aug_fn(image)

            new_mask_name = f"{mask_name.split('.')[0]}_{aug_name}.png"
            new_img_name = f"{mask_name.split('.')[0]}_{aug_name}.png"

            new_mask.save(os.path.join(new_mask_dir, new_mask_name))
            new_img.save(os.path.join(new_img_dir, new_img_name))

print("✅ ¡Aumento de datos completado! Nuevas imágenes y máscaras guardadas.")