import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import UNet  # Importar tu modelo U-Net
from utils.colormap import land_cover_cmap 
from matplotlib.patches import Patch

# Configuración
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 12
model_path = "model_99_0.3185.pth"
test_image_path = "train/img300/tile_7200_19200.png"

# Cargar el modelo entrenado
model = UNet(num_classes=num_classes, image_size= 300).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Transformaciones de imagen (deben coincidir con las usadas en entrenamiento)
transform = transforms.Compose([
    # transforms.Resize((256, 256)), # No es correcto
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1344, 0.1254, 0.0772], std=[0.1112, 0.0871, 0.0663])
])

# Función para realizar predicción en una imagen
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    # Convertir la salida a máscara de clases
    predicted_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    return image, predicted_mask

# Función para visualizar la imagen, la predicción y la leyenda de clases detectadas
def visualize_prediction(image_path, threshold=0.05):
    image, predicted_mask = predict(image_path)

    # Calcular la frecuencia de cada clase en la máscara predicha
    unique_classes, counts = np.unique(predicted_mask, return_counts=True)
    total_pixels = predicted_mask.size
    class_frequencies = counts / total_pixels

    # Filtrar clases que aparecen más del umbral de porcentaje
    filtered_classes = unique_classes[class_frequencies > threshold]

    # Obtener colores y etiquetas correspondientes solo para las clases filtradas
    legend_patches = [
        Patch(color=np.array(land_cover_cmap(i / num_classes)))
        for i in filtered_classes
    ]

    # Convertimos la máscara a una imagen en color
    colored_mask = land_cover_cmap(predicted_mask / num_classes)

    # Mostrar la imagen y la predicción
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(image)
    axes[0].set_title("Imagen de Entrada")
    axes[0].axis("off")

    axes[1].imshow(colored_mask)
    axes[1].set_title("Predicción de Segmentación")
    axes[1].axis("off")

    # Agregar leyenda con las clases detectadas
    if legend_patches:
        fig.legend(handles=legend_patches, loc="upper right", title="", fontsize=10)

    plt.show()

# Ejecutar la predicción en la imagen de prueba
visualize_prediction(test_image_path, threshold=0.02)
