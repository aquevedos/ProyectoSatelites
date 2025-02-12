import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import UNet  # Importar tu modelo U-Net
from utils.colormap import land_cover_cmap  # Importar colormap personalizado

# Definir etiquetas de clases
class_labels = [
    "Fuera de los límites", "Cultivos herbáceos", "Huertos", "Viñedos", "Olivares",
    "Otros cultivos leñosos", "Cultivos en transformación", "Bosques densos de coníferas",
    "Bosques densos de frondosas", "Bosques densos de esclerófilas", "Matorral",
    "Bosques claros de coníferas", "Bosques claros de frondosas", "Bosques claros de esclerófilas",
    "Praderas y pastizales", "Bosques ribereños", "Suelo forestal desnudo", "Zonas quemadas",
    "Zonas rocosas", "Playas", "Humedales", "Zona urbana", "Eixample", "Áreas urbanas laxas",
    "Edificios aislados", "Áreas residenciales aisladas", "Zonas verdes", "Áreas industriales/comerciales",
    "Áreas deportivas y de ocio", "Minas o vertederos", "Áreas en transformación",
    "Red vial", "Suelo urbano desnudo", "Áreas aeroportuarias", "Red ferroviaria",
    "Áreas portuarias", "Embalses", "Lagos y lagunas", "Cursos de agua", "Balsas",
    "Canales artificiales", "Mar"
]

# Configuración
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 42
model_path = "./modelos/model_99_1.0819.pth"
test_image_path = "./test/tile_21504_10752.png"

# Cargar el modelo entrenado
model = UNet(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Transformaciones de imagen (deben coincidir con las usadas en entrenamiento)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
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

# Función para mostrar solo las clases detectadas y visualizar la imagen y la máscara
def show_detected_classes(image_path):
    image, predicted_mask = predict(image_path)
    
    # Extraer clases únicas detectadas en la predicción
    unique_classes = np.unique(predicted_mask)
    
    print("Clases detectadas en la imagen:")
    for class_id in unique_classes:
        print(f"Clase {class_id}: {class_labels[class_id]}")
    
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

    plt.show()

# Ejecutar la predicción en la imagen de prueba
show_detected_classes(test_image_path)