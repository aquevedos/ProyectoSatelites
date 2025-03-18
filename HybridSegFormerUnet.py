import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import SegformerModel, SegformerConfig
from sklearn.metrics import confusion_matrix, accuracy_score

IMG_SIZE = 256          
NUM_CLASSES = 12       
BATCH_SIZE = 8
EPOCHS = 100
LR = 6e-5

IMG_DIR = "filtered_images"
MASK_DIR = "filtered_masks"

def load_image_or_mask(path, is_mask=False):
    img = Image.open(path).convert("L" if is_mask else "RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), Image.NEAREST if is_mask else Image.BILINEAR)
    img = np.array(img)

    if is_mask:
        return torch.tensor(img, dtype=torch.long)
    else:
        img = img.transpose(2, 0, 1)  
        return torch.tensor(img, dtype=torch.float32) / 255.0
        
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.tif')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.tif')])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        image = load_tiff_image(img_path, is_mask=False)  
        mask = load_tiff_image(mask_path, is_mask=True)    
        return image, mask

# DataLoaders
full_dataset = SegmentationDataset(IMG_DIR, MASK_DIR)
split = int(0.8 * len(full_dataset))
train_dataset = torch.utils.data.Subset(full_dataset, list(range(split)))
val_dataset = torch.utils.data.Subset(full_dataset, list(range(split, len(full_dataset))))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True):
        super(DecoderBlock, self).__init__()
        self.upsample = upsample
        layers = []
        if self.upsample:
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)

class HybridSegFormerUNet(nn.Module):
    """
    Hybrid model combining a SegFormer encoder with a UNet-style decoder.
    For "nvidia/mit-b0", the encoder hidden states are at:
      - f1: (B, 32, H/4, W/4)
      - f2: (B, 64, H/8, W/8)
      - f3: (B, 160, H/16, W/16)
      - f4: (B, 256, H/32, W/32)
    """
    def __init__(self, num_classes):
        super(HybridSegFormerUNet, self).__init__()
        config = SegformerConfig.from_pretrained("nvidia/mit-b0")
        config.output_hidden_states = True
        self.encoder = SegformerModel.from_pretrained("nvidia/mit-b0", config=config)

        self.decoder4 = DecoderBlock(in_channels=256, out_channels=160, upsample=True)    
        self.decoder3 = DecoderBlock(in_channels=160 + 160, out_channels=64, upsample=True) 
        self.decoder2 = DecoderBlock(in_channels=64 + 64, out_channels=32, upsample=True)   
        self.decoder1 = DecoderBlock(in_channels=32 + 32, out_channels=32, upsample=False)  

        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def forward(self, x):
        encoder_outputs = self.encoder(pixel_values=x, output_hidden_states=True)
        hidden_states = encoder_outputs.hidden_states
        f1, f2, f3, f4 = hidden_states

        d4 = self.decoder4(f4)                           
        d3 = self.decoder3(torch.cat([d4, f3], dim=1))     
        d2 = self.decoder2(torch.cat([d3, f2], dim=1))     
        d1 = self.decoder1(torch.cat([d2, f1], dim=1))     
        logits = self.final_conv(d1)                      
        # Upsample logits to match the original input resolution (256x256)
        logits = F.interpolate(logits, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        return logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridSegFormerUNet(num_classes=NUM_CLASSES).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=255)

best_val_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images = images.to(device)   
        masks = masks.to(device)     
        
        optimizer.zero_grad()
        outputs = model(images)      
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(outputs, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())
    val_loss /= len(val_loader)
    
    # Accuracy
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    class_acc = []
    for c in range(NUM_CLASSES):
        mask = (all_targets == c)
        if np.any(mask):
            acc = accuracy_score(all_targets[mask], all_preds[mask])
            class_acc.append(acc)
        else:
            class_acc.append(0.0)
    
    print(f"Epoch {epoch+1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    print("Class-wise Accuracy:", [f"{acc:.2%}" for acc in class_acc])

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), f"hybrid_segformer_unet_best_epoch{epoch+1}.pth")
        print("Saved best model!")

# Confusion matrix
cm = confusion_matrix(all_targets.flatten(), all_preds.flatten(), labels=np.arange(NUM_CLASSES))
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion matrix")
plt.show()
