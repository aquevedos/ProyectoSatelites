import os
import cv2
import numpy as np

'''
This script calculates the mean and standard deviation of images in a given folder.
'''

image_folder = './train/img'

sum_pixels = np.zeros(3)
sum_squared_pixels = np.zeros(3)
total_pixels = 0

for img_name in os.listdir(image_folder):
    img_path = os.path.join(image_folder, img_name)
    
    if img_name.endswith(('.png', '.jpg', '.jpeg')):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32) / 255.0
        
        sum_pixels += image.sum(axis=(0, 1))
        sum_squared_pixels += (image ** 2).sum(axis=(0, 1))
        
        total_pixels += image.shape[0] * image.shape[1]

mean = sum_pixels / total_pixels
std = np.sqrt((sum_squared_pixels / total_pixels) - (mean ** 2))

print(f"Calculated mean: {mean}")  
print(f"Calculated standard deviation: {std}")  