import os
import cv2
import numpy as np

# Directorio donde están las imágenes
image_folder = 'D:\data\modeloData\img'

# Variables para calcular media y desviación estándar
sum_pixels = np.zeros(3)
sum_squared_pixels = np.zeros(3)
total_pixels = 0

# Iterar sobre todas las imágenes
for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    
    if img_name.endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imread(img_path)  # Leer con OpenCV (BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir a RGB

        # Normalizar a rango [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Acumular valores por canal
        sum_pixels += image.sum(axis=(0, 1))
        sum_squared_pixels += (image ** 2).sum(axis=(0, 1))
        
        # Contar total de píxeles por imagen
        total_pixels += image.shape[0] * image.shape[1]

# Calcular la media y desviación estándar
mean = sum_pixels / total_pixels
std = np.sqrt((sum_squared_pixels / total_pixels) - (mean ** 2))

print(f"Media calculada (OpenCV): {mean}")
print(f"Desviación estándar calculada (OpenCV): {std}")