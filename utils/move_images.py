import os
import shutil
import random

'''
This script organizes images and their corresponding masks into training and testing folders. 
It ensures that each image has a corresponding mask before splitting the dataset. 
80% of the images (and their masks) are randomly selected for training, while the rest are used for testing.
'''

source_img_folder = "./img/"
source_mask_folder = "./mask/"
train_img_folder = "./train/img/"
train_mask_folder = "./train/mask/"

os.makedirs(train_img_folder, exist_ok=True)
os.makedirs(train_mask_folder, exist_ok=True)

image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif"}
images = [f for f in os.listdir(source_img_folder) if os.path.splitext(f)[1].lower() in image_extensions]
masks = [f for f in os.listdir(source_mask_folder) if os.path.splitext(f)[1].lower() == ".png"]

# Ensure images and masks match
image_names = set(os.path.splitext(f)[0] for f in images)
mask_names = set(os.path.splitext(f)[0] for f in masks)
print(len(image_names))
print(len(mask_names))

if image_names != mask_names:
    print("Some images do not have a corresponding mask or vice versa.")
    exit(1)

num_images_to_copy = int(len(images) * 0.8)

selected_images = random.sample(images, num_images_to_copy)

# Copy selected images and masks to the training folder
for image in selected_images:
    img_name = os.path.splitext(image)[0]
    mask = img_name + ".png"

    src_img_path = os.path.join(source_img_folder, image)
    dst_img_path = os.path.join(train_img_folder, image)
    shutil.copy(src_img_path, dst_img_path)

    src_mask_path = os.path.join(source_mask_folder, mask)
    dst_mask_path = os.path.join(train_mask_folder, mask)
    shutil.copy(src_mask_path, dst_mask_path)

# Copy remaining images to the test folder
remaining_images = [f for f in images if f not in selected_images]

for image in remaining_images:
    src_img_path = os.path.join(source_img_folder, image)
    shutil.copy(src_img_path)

print(f"{num_images_to_copy} images and their masks have been copied to the training folder.")
print("The remaining images have been copied to the test folder.")