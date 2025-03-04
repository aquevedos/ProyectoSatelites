import os

# Directorio de las imágenes
IMG_DIR = './train/img'

# Lista de archivos no .png
img_files = os.listdir(IMG_DIR)
non_png_files = [f for f in img_files if not f.endswith('.png')]

# Eliminar archivos no .png
for file in non_png_files:
    file_path = os.path.join(IMG_DIR, file)
    os.remove(file_path)  # Eliminar archivo
    print(f"Archivo eliminado: {file_path}")