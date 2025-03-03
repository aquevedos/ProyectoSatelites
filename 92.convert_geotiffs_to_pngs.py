import os
import rasterio
from rasterio.plot import reshape_as_image
import matplotlib.pyplot as plt

# Paths
input_dir = "d:\\geotiffs_tots"  # Directory containing 1000x1000 GeoTIFF files
output_dir = "d:\\geotiffs_tots\\pngs_transformed"  # Directory to save PNG files

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through all GeoTIFF files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".tif") or filename.endswith(".tiff"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".png")

        # Open the GeoTIFF file
        with rasterio.open(input_path) as src:
            # Read the data
            data = src.read()  # Reads all bands
            profile = src.profile

            # If there's more than one band, convert to RGB (assumes 3 bands minimum)
            if data.shape[0] >= 3:
                rgb = data[:3]  # Use the first three bands
                rgb = reshape_as_image(rgb)  # Reshape for plotting
            else:
                # For single-band images, normalize to [0, 255] for visualization
                band = data[0]
                band_min, band_max = band.min(), band.max()
                rgb = 255 * (band - band_min) / (band_max - band_min)
                rgb = rgb.astype("uint8")

            # Save as PNG using matplotlib
            plt.imsave(output_path, rgb, cmap="gray" if data.shape[0] == 1 else None)

        print(f"Converted {input_path} to {output_path}")

print(f"All GeoTIFF files have been converted to PNG and saved in {output_dir}")
