import os
import shutil
import random

# Configuración de las carpetas
source_img_folder = "./img/"
source_mask_folder = "./mask/"
train_img_folder = "./train/img/"
train_mask_folder = "./train/mask/"

# Crear las carpetas de destino si no existen
os.makedirs(train_img_folder, exist_ok=True)
os.makedirs(train_mask_folder, exist_ok=True)

# Listar todas las imágenes y máscaras en sus respectivas carpetas
image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif"}
images = [f for f in os.listdir(source_img_folder) if os.path.splitext(f)[1].lower() in image_extensions]
masks = [f for f in os.listdir(source_mask_folder) if os.path.splitext(f)[1].lower() == ".png"]  # Asumiendo que las máscaras son .tif

# Verificar que las imágenes y máscaras coinciden
image_names = set(os.path.splitext(f)[0] for f in images)
mask_names = set(os.path.splitext(f)[0] for f in masks)
print(len(image_names))
print(len(mask_names))
# Verificar que todas las imágenes tengan su correspondiente máscara
if image_names != mask_names:
    print("Algunas imágenes no tienen su correspondiente máscara o viceversa.")
    exit(1)

# Calcular el número de imágenes a copiar
num_images_to_copy = int(len(images) * 0.8)

# Seleccionar aleatoriamente las imágenes a copiar
selected_images = random.sample(images, num_images_to_copy)

# Copiar las imágenes y máscaras seleccionadas
for image in selected_images:
    img_name = os.path.splitext(image)[0]  # Nombre de la imagen sin extensión
    mask = img_name + ".png"  # La máscara correspondiente

    # Copiar la imagen a la carpeta de entrenamiento
    src_img_path = os.path.join(source_img_folder, image)
    dst_img_path = os.path.join(train_img_folder, image)
    shutil.copy(src_img_path, dst_img_path)

    # Copiar la máscara a la carpeta de entrenamiento
    src_mask_path = os.path.join(source_mask_folder, mask)
    dst_mask_path = os.path.join(train_mask_folder, mask)
    shutil.copy(src_mask_path, dst_mask_path)

# Copiar las imágenes restantes a la carpeta de prueba
remaining_images = [f for f in images if f not in selected_images]

for image in remaining_images:
    # Copiar la imagen a la carpeta de prueba
    src_img_path = os.path.join(source_img_folder, image)
    shutil.copy(src_img_path)

print(f"Se han copiado {num_images_to_copy} imágenes y sus máscaras a la carpeta de entrenamiento.")
print(f"El resto de las imágenes se han copiado a la carpeta de prueba.")