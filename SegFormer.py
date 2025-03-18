import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerForSemanticSegmentation
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from PIL import Image
import torch.nn.functional as F

IMG_HEIGHT, IMG_WIDTH = 256, 256
NUM_CLASSES = 13
BATCH_SIZE = 8
EPOCHS = 10
LR = 6e-5

# Directoris d'imatges
IMG_DIR1 = "train/img300"
IMG_DIR2 = "new_images"

# Directoris de màscares
MASK_DIR1 = "train/mask300"
MASK_DIR2 = "new_masks"

def load_image_or_mask(path, is_mask=False):
    img = Image.open(path).convert("L" if is_mask else "RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST if is_mask else Image.BILINEAR)
    img = np.array(img)

    if is_mask:
        return torch.tensor(img, dtype=torch.long)
    else:
        img = img.transpose(2, 0, 1)  
        return torch.tensor(img, dtype=torch.float32) / 255.0  

def get_image_mask_pairs(img_dir1, img_dir2, mask_dir1, mask_dir2):
    img_files = sorted([os.path.join(img_dir1, f) for f in os.listdir(img_dir1) if f.endswith(".png")])
    img_files += sorted([os.path.join(img_dir2, f) for f in os.listdir(img_dir2) if f.endswith(".png")])

    mask_files = sorted([os.path.join(mask_dir1, f) for f in os.listdir(mask_dir1) if f.endswith(".png")])
    mask_files += sorted([os.path.join(mask_dir2, f) for f in os.listdir(mask_dir2) if f.endswith(".png")])

    return img_files, mask_files

image_files, mask_files = get_image_mask_pairs(IMG_DIR1, IMG_DIR2, MASK_DIR1, MASK_DIR2)

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = load_image_or_mask(self.image_paths[idx], is_mask=False)
        mask = load_image_or_mask(self.mask_paths[idx], is_mask=True)
        return img, mask

split = int(0.8 * len(image_files))
train_dataset = SegmentationDataset(image_files[:split], mask_files[:split])
val_dataset = SegmentationDataset(image_files[split:], mask_files[split:])

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)
criterion = CrossEntropyLoss(ignore_index=255)

best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for images_batch, masks_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
        outputs = model(images_batch, labels=masks_batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    model.eval()
    val_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images_batch, masks_batch in val_loader:
            images_batch, masks_batch = images_batch.to(device), masks_batch.to(device)
            outputs = model(images_batch)
            logits = outputs.logits
            logits = F.interpolate(logits, size=masks_batch.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(logits, masks_batch)
            val_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(masks_batch.cpu())

    val_loss /= len(val_loader)
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()

    class_acc = []
    for c in range(NUM_CLASSES):
        mask = (all_targets == c)
        acc = accuracy_score(all_targets[mask], all_preds[mask]) if np.any(mask) else 0.0
        class_acc.append(acc)

    print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print("Class-wise Accuracy:", [f"{acc:.2%}" for acc in class_acc])

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"segformer_best_epoch{epoch+1}.pth")
        print("Saved best model!")

cm = confusion_matrix(all_targets.flatten(), all_preds.flatten(), labels=np.arange(NUM_CLASSES))
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
