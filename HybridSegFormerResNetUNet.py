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
import torchvision.models as models
from PIL import Image
from torchview import draw_graph

IMG_SIZE = 256          
NUM_CLASSES = 13        
BATCH_SIZE = 8
EPOCHS = 2
LR = 6e-5
IMG_DIR = "filtered_images"
MASK_DIR = "filtered_masks"

def load_image_or_mask(path, is_mask=False):
    if is_mask:
        # Load TIFF mask
        with rasterio.open(path) as src:
            img = src.read()
            img = img[0, :, :]  # Take first band
            img = np.expand_dims(img, axis=0)  # Add channel dim
            tensor_img = torch.tensor(img).unsqueeze(0)  # Add batch dim
            tensor_img = F.interpolate(tensor_img.float(), size=(IMG_SIZE, IMG_SIZE), mode='nearest')
            tensor_img = tensor_img.squeeze(0).squeeze(0)  # Remove dims
            return tensor_img.long()
    else:
        # Load JPG image
        img = Image.open(path).convert('RGB')
        img = np.array(img)  # (H, W, 3)
        img = img.transpose(2, 0, 1)  # (3, H, W)
        tensor_img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        tensor_img = F.interpolate(tensor_img, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        tensor_img = tensor_img.squeeze(0) / 255.0  # Normalize
        return tensor_img

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
        image = load_image_or_mask(img_path, is_mask=False)  
        mask = load_image_or_mask(mask_path, is_mask=True)    
        return image, mask

# DataLoaders
full_dataset = SegmentationDataset(IMG_DIR, MASK_DIR)
split = int(0.8 * len(full_dataset))
train_dataset = torch.utils.data.Subset(full_dataset, list(range(split)))
val_dataset = torch.utils.data.Subset(full_dataset, list(range(split, len(full_dataset))))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

class ResNet34Encoder(nn.Module):
    def __init__(self):
        super(ResNet34Encoder, self).__init__()
        resnet = models.resnet34(pretrained=True)
        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  
        self.layer2 = resnet.layer2  
        self.layer3 = resnet.layer3  
        self.layer4 = resnet.layer4  
    def forward(self, x):
        x = self.initial(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return f1, f2, f3, f4

class FusionBlock(nn.Module):
    """
    Es fusionen els 2 feature maps concatenant-los i s'aplica una conv 1x1 per reduir el nombre de canals.
    """
    def __init__(self, in_channels, out_channels):
        super(FusionBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)

# ResNet34 + SegFormer
class HybridEncoder(nn.Module):
    """
    Aqui es processa el input utilitzant la ResNet34 i una branch del SegFormer. Queda visualitzat en el gràfic que es
    mostra al executar el codi.
      SegFormer (nvidia/mit-b0):
        f1: (B, 32, 64, 64)
        f2: (B, 64, 32, 32)
        f3: (B, 160, 16, 16)
        f4: (B, 256, 8, 8)
    A la part de Fusion, es concatenen els channels i es fa una conv 1x1, per a reduir les dimensions.
    """
    def __init__(self):
        super(HybridEncoder, self).__init__()
        self.resnet_encoder = ResNet34Encoder()
        
        config = SegformerConfig.from_pretrained("nvidia/mit-b0")
        config.output_hidden_states = True
        self.segformer = SegformerModel.from_pretrained("nvidia/mit-b0", config=config)

        self.fuse1 = FusionBlock(64 + 32, 64)    
        self.fuse2 = FusionBlock(128 + 64, 128)    
        self.fuse3 = FusionBlock(256 + 160, 256)   
        self.fuse4 = FusionBlock(512 + 256, 512)   
        
    def forward(self, x):
        res_f1, res_f2, res_f3, res_f4 = self.resnet_encoder(x)
        seg_outputs = self.segformer(pixel_values=x, output_hidden_states=True)
        seg_hidden = seg_outputs.hidden_states  
        seg_f1, seg_f2, seg_f3, seg_f4 = seg_hidden

        fused1 = self.fuse1(torch.cat([res_f1, seg_f1], dim=1))  
        fused2 = self.fuse2(torch.cat([res_f2, seg_f2], dim=1))  
        fused3 = self.fuse3(torch.cat([res_f3, seg_f3], dim=1))  
        fused4 = self.fuse4(torch.cat([res_f4, seg_f4], dim=1))  
        return fused1, fused2, fused3, fused4

class DecoderBlock(nn.Module):
    """
    UNet-style decoder
    """
    def __init__(self, in_channels, out_channels, upsample=True):
        super(DecoderBlock, self).__init__()
        self.upsample = upsample
        layers = []
        if upsample:
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


class HybridDecoder(nn.Module):
    def __init__(self, num_classes):
        super(HybridDecoder, self).__init__()
        self.decoder4 = DecoderBlock(in_channels=512, out_channels=256, upsample=True)   
        self.decoder3 = DecoderBlock(in_channels=512, out_channels=256, upsample=True)   
        self.decoder2 = DecoderBlock(in_channels=384, out_channels=128, upsample=True)   
        self.decoder1 = DecoderBlock(in_channels=192, out_channels=64, upsample=True)    
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, fused1, fused2, fused3, fused4):
        d4 = self.decoder4(fused4)                            
        d3 = self.decoder3(torch.cat([d4, fused3], dim=1))       
        d2 = self.decoder2(torch.cat([d3, fused2], dim=1))       
        d1 = self.decoder1(torch.cat([d2, fused1], dim=1))       
        logits = self.final_conv(d1)                           
        logits = F.interpolate(logits, scale_factor=2, mode='bilinear', align_corners=False)
        return logits

# ResNet34 + SegFormer Encoder, UNet-style Decoder
class HybridResNetSegFormer(nn.Module):
    def __init__(self, num_classes):
        super(HybridResNetSegFormer, self).__init__()
        self.encoder = HybridEncoder()
        self.decoder = HybridDecoder(num_classes)
        
    def forward(self, x):
        fused1, fused2, fused3, fused4 = self.encoder(x)
        logits = self.decoder(fused1, fused2, fused3, fused4)
        return logits


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridResNetSegFormer(num_classes=NUM_CLASSES).to(device)
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
    
    # Validació
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
        torch.save(model.state_dict(), f"hybrid_resnet_segformer_best_epoch{epoch+1}.pth")
        print("Saved best model!")

# Confusion matrix
cm = confusion_matrix(all_targets.flatten(), all_preds.flatten(), labels=np.arange(NUM_CLASSES))
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion matrix")
plt.show()

# Visualització
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridResNetSegFormer(num_classes=NUM_CLASSES).to(device)
    dummy_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

    model_graph = draw_graph(
        model,
        input_data=dummy_input,
        graph_name="HybridModel",
        expand_nested=True,  
        depth=2,             
    )
    save_path = os.path.abspath("hybrid_layers")  
    model_graph.visual_graph.render(save_path, format="png", cleanup=True)
    print("Diagrama guardat a:", save_path + ".png")