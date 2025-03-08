import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from utils.dataset import SegmentationDataset, get_transforms
from model import UNet
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Función de entrenamiento
def train():
    best_loss = float('inf')  # Para rastrear la mejor pérdida

    for epoch in range(epochs):
        epoch_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model.train()
        running_loss = 0.0
        class_correct = torch.zeros(num_classes).to(device)
        class_total = torch.zeros(num_classes).to(device)

        all_preds = []
        all_labels = []

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            # No modificar la clase 41, solo ignorar la 0 en la pérdida
            optimizer.zero_grad()
            outputs = model(images)

            # Calcular pérdida
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calcular accuracy por clase (sin contar clase 0)
            _, predicted = torch.max(outputs, 1)
            for i in range(num_classes):
                if i != 0:  # Ignorar solo la clase 0
                    class_correct[i] += (predicted[masks == i] == i).sum().item()
                    class_total[i] += (masks == i).sum().item()

            # Collect all predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_labels.extend(masks.cpu().numpy().flatten())

            # Calculate IoU and Dice for each class
            iou_list = calculate_iou(predicted, masks, num_classes)
            dice_list = calculate_dice(predicted, masks, num_classes)

            # Actualizar barra de progreso
            loop.set_postfix(loss=loss.item())

        # Epoch finalizado
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
        print(f'Epoch started at {epoch_start_time} and ended at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

        for i in range(num_classes):
            if i != 0 and class_total[i] > 0:  # Mostrar accuracy para todas las clases excepto la 0
                acc = 100 * class_correct[i] / class_total[i]
                print(f"Clase {i}: {acc:.2f}% accuracy")
                print(f"Clase {i}: IoU: {iou_list[i-1]:.4f}, Dice: {dice_list[i-1]:.4f}")

        # Ajustar el learning rate si la loss no mejora
        scheduler.step(epoch_loss)

        # Guardar el mejor modelo
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            name = f"model_{epoch}_{epoch_loss:.4f}.pth"
            torch.save(model.state_dict(), name)
            print(f"Mejor modelo guardado con Loss: {epoch_loss:.4f}")

        # Calculate and display confusion matrix in the last epoch
        if epoch == epochs - 1:
            cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(range(num_classes)), yticklabels=list(range(num_classes)))
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix for Epoch {epoch+1}')
            plt.show()
        

def calculate_iou(pred, target, num_classes):
    iou_list = []
    for cls in range(1, num_classes):  # Ignoring class 0
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            iou_list.append(float('nan'))  # If no ground truth, do not include in IoU calculation
        else:
            iou_list.append(intersection / union)
    return iou_list

def calculate_dice(pred, target, num_classes):
    dice_list = []
    for cls in range(1, num_classes):  # Ignoring class 0
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum().item()
        denominator = pred_inds.sum().item() + target_inds.sum().item()
        if denominator == 0: # To avoid division by zero
            dice = 0
        else:
            dice = (2 * intersection) / denominator
        dice_list.append(dice)
    return dice_list

if __name__ == "__main__":

    # GPU CON CUDA y MPS para macintosh con chips Apple Silicon
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print (f"Device utilizado: {device}")

    num_classes = 12
    batch_size = 12
    epochs = 100
    lr = 0.001
    workers = 5
    pin_memory = True

    # Dataset y DataLoader
    train_images_dir = "train/img300"
    train_masks_dir = "train/mask300"
    train_dataset = SegmentationDataset(train_images_dir, train_masks_dir, transform=get_transforms())

    train_size = int(0.8 * len(train_dataset)) 
    val_size = len(train_dataset) - train_size 
    print(f"Train size images: {train_size}, Validation size images: {val_size}")

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory) 

    # Normalización de pesos para la función de pérdida (ignorando solo la clase 0)
    class_weights = torch.ones(num_classes).to(device)
    class_weights[0] = 0  # Ignorar clase 0 ("out of borders")

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)  # Ignoramos solo la clase 0

    # Modelo y optimizador
    model = UNet(num_classes=num_classes, image_size=300).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Scheduler para reducir el learning rate si la pérdida no mejora
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) # Verbose está deprecado

    train()