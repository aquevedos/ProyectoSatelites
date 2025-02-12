import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import SegmentationDataset, get_transforms
from model import UNet
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configuración
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 42
batch_size = 8
epochs = 200
lr = 0.001

# Dataset y DataLoader
train_images_dir = "train/img"
train_masks_dir = "train/mask"
train_dataset = SegmentationDataset(train_images_dir, train_masks_dir, transform=get_transforms())
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Normalización de pesos para la función de pérdida (ignorando clases 0 y 41)
class_weights = torch.ones(num_classes).to(device)
class_weights[0] = 0  # Clase "mar" ignorada
class_weights[41] = 0  # Clase "out of borders" ignorada
criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)  # Ignoramos la clase 0

# Modelo y optimizador
model = UNet(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Scheduler para reducir el learning rate si la pérdida no mejora
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Función de entrenamiento
def train():
    best_loss = float('inf')  # Para rastrear la mejor pérdida

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        class_correct = torch.zeros(num_classes).to(device)
        class_total = torch.zeros(num_classes).to(device)

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            # Convertir la clase 41 en 0 para que sea ignorada
            masks[masks == 41] = 0  

            optimizer.zero_grad()
            outputs = model(images)

            # Calcular pérdida
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calcular accuracy por clase (sin contar clases 0 y 41)
            _, predicted = torch.max(outputs, 1)
            for i in range(num_classes):
                if i not in [0, 41]:  # Ignorar clases no deseadas
                    class_correct[i] += (predicted[masks == i] == i).sum().item()
                    class_total[i] += (masks == i).sum().item()

            # Actualizar barra de progreso
            loop.set_postfix(loss=loss.item())

        # Resultados de la época
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        
        for i in range(num_classes):
            if i not in [0, 41] and class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f"Clase {i}: {acc:.2f}% accuracy")

        # Ajustar el learning rate si la loss no mejora
        scheduler.step(epoch_loss)

        # Guardar el mejor modelo
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            name = f"model_{epoch}_{epoch_loss:.4f}.pth"
            torch.save(model.state_dict(), name)
            print(f"Mejor modelo guardado con Loss: {epoch_loss:.4f}")

train()