import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from utils.colormap import land_cover_cmap

'''
This script performs data augmentation on image-mask pairs for semantic segmentation.
It applies a series of transformations (rotation and flipping) to both the images and their corresponding masks. 
The augmentation is performed only on images where the associated mask contains a sufficiently under-represented class.
'''

img_dir = "./train/img"
mask_dir = "./train/mask"
new_img_dir = "./new_images"
new_mask_dir = "./new_masks"

os.makedirs(new_img_dir, exist_ok=True)
os.makedirs(new_mask_dir, exist_ok=True)

# Under-represented classes
low_rep_classes = {5, 7, 8, 10, 11}
threshold = 0.05 

augmentations = {
    "rot90": lambda x: x.rotate(90),
    "rot180": lambda x: x.rotate(180),
    "rot270": lambda x: x.rotate(270),
    "flip_h": lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
    "flip_v": lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
}

# Convert colourmap to array for comparison
color_palette = np.array([np.round(np.array(land_cover_cmap(i)[:3]) * 255).astype(int) for i in range(len(land_cover_cmap.colors))])

def rgb_to_class(mask_rgb):
    mask_array = np.array(mask_rgb)
    class_mask = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.uint8)

    for class_idx, color in enumerate(color_palette):
        matches = np.all(mask_array == color, axis=-1)
        class_mask[matches] = class_idx 

    return class_mask

def analyze_mask(mask_array):
    total_pixels = mask_array.size
    class_counts = {cls: np.sum(mask_array == cls) / total_pixels for cls in np.unique(mask_array)}
    return class_counts

for mask_name in tqdm(os.listdir(mask_dir), desc="Procesando máscaras"):
    mask_path = os.path.join(mask_dir, mask_name)
    img_path = os.path.join(img_dir, mask_name)
    
    if not os.path.exists(img_path):
        continue
    
    mask_rgb = Image.open(mask_path).convert("RGB")
    mask_array = rgb_to_class(mask_rgb)
    class_distribution = analyze_mask(mask_array)

    # Check whether any of the minority classes are sufficiently represented.
    if any(class_distribution.get(cls, 0) > threshold for cls in low_rep_classes):
        image = Image.open(img_path).convert("RGB")
        
        # Apply each transformation and save
        for aug_name, aug_fn in augmentations.items():
            new_mask = aug_fn(mask_rgb)
            new_img = aug_fn(image)

            new_mask_name = f"{mask_name.split('.')[0]}_{aug_name}.png"
            new_img_name = f"{mask_name.split('.')[0]}_{aug_name}.png"

            new_mask.save(os.path.join(new_mask_dir, new_mask_name))
            new_img.save(os.path.join(new_img_dir, new_img_name))

print("✅ ¡Data augmentation completed! New images and masks saved.")