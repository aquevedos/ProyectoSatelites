import os
import torch
import numpy as np
import cv2
import rasterio
import matplotlib.pyplot as plt

# Define a colormap for the 12 classes
colormap = np.array([
    [0, 0, 0],       # Background (or NoData if applicable)
    [255, 0, 0],     # Class 1
    [0, 255, 0],     # Class 2
    [0, 0, 255],     # Class 3
    [255, 255, 0],   # Class 4
    [255, 0, 255],   # Class 5
    [0, 255, 255],   # Class 6
    [128, 128, 0],   # Class 7
    [128, 0, 128],   # Class 8
    [0, 128, 128],   # Class 9
    [128, 128, 128], # Class 10
    [64, 0, 64],     # Class 11
    [192, 192, 192], # Class 12
])

def colorize_mask(mask):
    """Convert a mask of class indices into an RGB image using the colormap."""
    color_mask = colormap[mask]
    return color_mask.astype(np.uint8)

def predict_and_visualize(model, image_path, mask_path, device):
    """Loads an image and mask, performs prediction, and visualizes results."""
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize and reorder channels
    image_tensor = image_tensor.unsqueeze(0).to(device)  # Add batch dimension

    # Load ground truth mask
    with rasterio.open(mask_path) as mask_src:
        mask = mask_src.read(1)  # Read the single-band mask
    mask = np.clip(mask - 1, 0, 11)  # Convert from [1,12] to [0,11] for compatibility with predictions
    mask_colored = colorize_mask(mask)

    # Perform prediction
    model.eval()
    with torch.no_grad():
        pred = model(image_tensor)
        pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()  # Convert to class indices

    pred_colored = colorize_mask(pred)

    # Plot the results
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[1].imshow(mask_colored)
    ax[1].set_title("Ground Truth Mask")
    ax[2].imshow(pred_colored)
    ax[2].set_title("Predicted Mask")
    
    for a in ax:
        a.axis("off")

    plt.show()



BASE_OUTPUT = r'G:\My Drive\Personal\PostgrauAIDL\Proyecto-Grupo\github-ICGC\CatLC\esteve\resultados'
MODEL_PATH = os.path.join(BASE_OUTPUT, "unet_satelitales.pth")
# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.load(MODEL_PATH).to(device)  # Update with correct path

# List of test images and masks
test_images = ["path/to/test_image_1.png", "path/to/test_image_2.png", "path/to/test_image_3.png"]  # Update paths
test_masks = ["path/to/test_mask_1.tif", "path/to/test_mask_2.tif", "path/to/test_mask_3.tif"]  # Update paths

# Run predictions and visualize results
for img_path, mask_path in zip(test_images, test_masks):
    predict_and_visualize(model, img_path, mask_path, device)
