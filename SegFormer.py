import os
import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils import resample

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
IMG_HEIGHT, IMG_WIDTH = 256, 256
IMG_CHANNELS = 3
NUM_CLASSES = 13
BATCH_SIZE = 8
EPOCHS = 10
LR = 6e-5

IMG_DIR = "filtered_images"
MASK_DIR = "filtered_masks"

# -----------------------------------------------------------------------------
# Data Loading and Preprocessing
# -----------------------------------------------------------------------------
def load_tiff_image(image_path, is_mask=False):
    with rasterio.open(image_path) as src:
        img = src.read()
        if is_mask:
            # For masks, assume single-band data: shape (H, W)
            img = img[0, :, :]  # shape: (H, W)
            img = np.expand_dims(img, axis=0)  # shape: (1, H, W)
            # Convert to tensor and add batch dimension -> (1, 1, H, W)
            tensor_img = torch.tensor(img)
            tensor_img = tensor_img.unsqueeze(0)
            # Resize mask using nearest mode (to preserve label values)
            tensor_img = torch.nn.functional.interpolate(tensor_img, size=(IMG_HEIGHT, IMG_WIDTH), mode='nearest')
            # Remove batch and channel dimensions -> (H, W)
            tensor_img = tensor_img.squeeze(0).squeeze(0)
            return tensor_img.long()
        else:
            # For images, ensure we have three channels.
            num_bands = img.shape[0]
            if num_bands >= 3:
                img = img[[2, 1, 0], :, :]  # rearrange to RGB (from BGR or similar)
            else:
                # If single channel, replicate it to create a 3-channel image.
                img = np.tile(img[0, :, :], (3, 1, 1))
            # Now img is in (C, H, W) format.
            tensor_img = torch.tensor(img).float()
            # Add batch dimension -> (1, C, H, W)
            tensor_img = tensor_img.unsqueeze(0)
            # Resize image using bilinear interpolation
            tensor_img = torch.nn.functional.interpolate(tensor_img, size=(IMG_HEIGHT, IMG_WIDTH), mode='bilinear', align_corners=False)
            # Remove batch dimension -> (C, H, W)
            tensor_img = tensor_img.squeeze(0)
            # Normalize to [0, 1]
            tensor_img /= 255.0
            return tensor_img

# Load images and masks into lists
image_files = sorted(os.listdir(IMG_DIR))
mask_files = sorted(os.listdir(MASK_DIR))
images, masks = [], []
for img_file, mask_file in zip(image_files, mask_files):
    img_path = os.path.join(IMG_DIR, img_file)
    mask_path = os.path.join(MASK_DIR, mask_file)
    img = load_tiff_image(img_path, is_mask=False)
    mask = load_tiff_image(mask_path, is_mask=True)
    # Ensure correct dimensions (images: [C, H, W], masks: [H, W])
    if img.shape[1:] == (IMG_HEIGHT, IMG_WIDTH) and mask.shape[-2:] == (IMG_HEIGHT, IMG_WIDTH):
        images.append(img)
        masks.append(mask)

# Stack into tensors
images = torch.stack(images)  # shape: (N, C, H, W)
masks = torch.stack(masks)    # shape: (N, H, W)

# -----------------------------------------------------------------------------
# Compute Class Distribution and Oversample Minority Cases
# -----------------------------------------------------------------------------
all_mask_values = masks.view(-1).numpy()
class_counts = np.bincount(all_mask_values, minlength=NUM_CLASSES)
print("Initial class distribution:", class_counts)

# For segmentation, oversampling is less straightforward than for classification.
# One common approach is to identify images that contain minority classes and oversample them.
def contains_minority(mask_tensor, threshold=50000):
    # Compute class counts for this image mask
    mask_np = mask_tensor.numpy()
    unique, counts = np.unique(mask_np, return_counts=True)
    for u, c in zip(unique, counts):
        if class_counts[u] < threshold:
            return True
    return False

# Separate indices for images with minority classes versus majority only
minority_indices = [i for i in range(len(masks)) if contains_minority(masks[i])]
majority_indices = [i for i in range(len(masks)) if i not in minority_indices]

# Oversample minority images so that their count matches the majority group size
if minority_indices:
    oversampled_minority = resample(minority_indices, replace=True, n_samples=len(majority_indices), random_state=42)
    indices = majority_indices + oversampled_minority
else:
    indices = list(range(len(masks)))

images = images[indices]
masks = masks[indices]
new_class_counts = np.bincount(masks.view(-1).numpy(), minlength=NUM_CLASSES)
print("Class distribution after oversampling:", new_class_counts)

# -----------------------------------------------------------------------------
# Train/Validation Split and Dataset Definition
# -----------------------------------------------------------------------------
split = int(0.8 * len(images))
train_images, val_images = images[:split], images[split:]
train_masks, val_masks = masks[:split], masks[split:]

class SegmentationDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]

train_dataset = SegmentationDataset(train_images, train_masks)
val_dataset = SegmentationDataset(val_images, val_masks)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# -----------------------------------------------------------------------------
# SegFormer Model Setup (Using Hugging Face Transformers)
# -----------------------------------------------------------------------------
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0",        # backbone architecture
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)
criterion = CrossEntropyLoss(ignore_index=255)

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for images_batch, masks_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images_batch = images_batch.to(device)  # (B, C, H, W)
        masks_batch = masks_batch.to(device)      # (B, H, W)
        
        # Forward pass with labels returns a dict with the loss
        outputs = model(images_batch, labels=masks_batch)
        loss = outputs.loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images_batch, masks_batch in val_loader:
            images_batch = images_batch.to(device)
            masks_batch = masks_batch.to(device)
            outputs = model(images_batch)
            logits = outputs.logits
            # Resize logits to the mask size if needed (typically not required if images are fixed)
            logits = torch.nn.functional.interpolate(
                logits, size=masks_batch.shape[-2:], mode='bilinear', align_corners=False
            )
            loss = criterion(logits, masks_batch)
            val_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(masks_batch.cpu())
    val_loss /= len(val_loader)
    
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    
    # Compute class-wise accuracy
    class_acc = []
    for c in range(NUM_CLASSES):
        mask = (all_targets == c)
        if np.any(mask):
            acc = accuracy_score(all_targets[mask], all_preds[mask])
            class_acc.append(acc)
        else:
            class_acc.append(0.0)
    
    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print("Class-wise Accuracy:", [f"{acc:.2%}" for acc in class_acc])
    
    # Save best model based on validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"segformer_best_epoch{epoch+1}.pth")
        print("Saved best model!")

# -----------------------------------------------------------------------------
# Confusion Matrix Visualization
# -----------------------------------------------------------------------------
cm = confusion_matrix(all_targets.flatten(), all_preds.flatten(), labels=np.arange(NUM_CLASSES))
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()