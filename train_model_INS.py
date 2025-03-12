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
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from collections import Counter

# GPU Selection
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device utilized: {device}")

num_classes = 12
batch_size = 12
epochs = 100
lr = 0.001
workers = 5
pin_memory = True
train_split_ratio = 0.8

# Dataset and DataLoader
train_images_dir = "train/img300"
train_masks_dir = "train/mask300"
train_dataset = SegmentationDataset(train_images_dir, train_masks_dir, transform=get_transforms())

# Splitting into train and validation sets
train_size = int(train_split_ratio * len(train_dataset))
val_size = len(train_dataset) - train_size
print(f"Train size images: {train_size}, Validation size images: {val_size}")

train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin_memory)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin_memory)

# Calculate class frequencies from dataset
print("Calculating class distribution...")
class_counts = torch.zeros(num_classes).to(device)

for _, masks in train_loader:
    masks = masks.to(device)
    for i in range(num_classes):
        class_counts[i] += (masks == i).sum().item()

# Compute Inverse Number of Samples (INS) class weights
class_weights = 1.0 / (class_counts + 1e-6)  # Avoid division by zero
class_weights = class_weights / class_weights.sum()  # Normalize weights
class_weights[0] = 0  # Ignore class 0 (out of borders)
print(f"Class Weights: {class_weights}")

# Apply INS-based weights to CrossEntropyLoss
criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)

# Model and optimizer
model = UNet(num_classes=num_classes, image_size=300).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Multi-GPU support
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs.")

# TensorBoard
writer = SummaryWriter()

# Training function
def train():
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        class_correct = torch.zeros(num_classes).to(device)
        class_total = torch.zeros(num_classes).to(device)

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Compute per-class accuracy (ignoring class 0)
            _, predicted = torch.max(outputs, 1)
            for i in range(num_classes):
                if i != 0:
                    class_correct[i] += (predicted[masks == i] == i).sum().item()
                    class_total[i] += (masks == i).sum().item()

            loop.set_postfix(loss=loss.item())

        # Epoch complete
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

        for i in range(1, num_classes):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f"Class {i}: {acc:.2f}% accuracy")

        scheduler.step(epoch_loss)

        # Save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f"model_{epoch}_{epoch_loss:.4f}.pth")
            print(f"Best model saved with Loss: {epoch_loss:.4f}")

        # TensorBoard Logging
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

train()

writer.close()
