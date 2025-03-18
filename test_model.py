import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from model import UNet
from utils.colormap import land_cover_cmap 
from matplotlib.patches import Patch

'''
This script uses a pre-trained U-Net model to perform image segmentation on a test image. 
It loads the model, applies transformations to the input image, predicts the segmentation mask, 
and then visualizes both the original image and the predicted segmentation mask with a color legend for the detected classes.
'''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 12
model_path = "model_99_0.3185.pth"
test_image_path = "train/img300/tile_7200_19200.png"

model = UNet(num_classes=num_classes, image_size= 300).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1344, 0.1254, 0.0772], std=[0.1112, 0.0871, 0.0663])
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

    predicted_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    return image, predicted_mask

def visualize_prediction(image_path, threshold=0.05):
    image, predicted_mask = predict(image_path)

    unique_classes, counts = np.unique(predicted_mask, return_counts=True)
    total_pixels = predicted_mask.size
    class_frequencies = counts / total_pixels

    filtered_classes = unique_classes[class_frequencies > threshold]

    legend_patches = [
        Patch(color=np.array(land_cover_cmap(i / num_classes)))
        for i in filtered_classes
    ]

    colored_mask = land_cover_cmap(predicted_mask / num_classes)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(image)
    axes[0].set_title("Input image")
    axes[0].axis("off")

    axes[1].imshow(colored_mask)
    axes[1].set_title("Segmentation forecasting")
    axes[1].axis("off")

    if legend_patches:
        fig.legend(handles=legend_patches, loc="upper right", title="", fontsize=10)

    plt.show()

visualize_prediction(test_image_path, threshold=0.02)
