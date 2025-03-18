import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from utils.colormap import land_cover_cmap

'''
This script defines a custom dataset class for semantic segmentation tasks.
The SegmentationDataset class loads images and their corresponding mask files, transforms them, and converts mask colors into class indices based on a predefined colormap.
It allows easy access to image-mask pairs, which are essential for training a segmentation model, such as U-Net.
'''

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir=None, transform=None, num_classes=12):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.num_classes = num_classes
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
        self.colormap = land_cover_cmap

    def __len__(self):
        return len(self.images)

    def color_to_class(self, mask):
        class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

        for i in range(self.num_classes):
            color = np.array(self.colormap(i)[:3])
            color = np.round(color * 255).astype(int)

            mask_pixels = np.all(np.abs(mask - color) <= 1, axis=-1)

            class_mask[mask_pixels] = i

        return class_mask

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        mask = np.array(mask) 

        class_mask = self.color_to_class(mask)

        class_mask = torch.tensor(class_mask, dtype=torch.long)

        return image, class_mask

def get_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1344, 0.1254, 0.0772], std=[0.1112, 0.0871, 0.0663])
    ])