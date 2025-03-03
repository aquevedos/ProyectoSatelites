<<<<<<< HEAD
"""
Autor: Esteve Graells

Utilitzo aquest script per a crear tiles en format PNG a partir de fitxers GeoTIFF de tres bandes.

Paràmetres:
    sentinel_file (str): Ruta al fitxer GeoTIFF d'entrada.
    output_dir (str): Directori on es guardaran les tessel·les individuals.
    subname (str): Subnom per a identificar les tessel·les.
"""

=======
>>>>>>> a60474462d71d79b45874a2a44ea096f812dd4ad
import os
import numpy as np

import rasterio
from rasterio.windows import Window
from rasterio.plot import show
from matplotlib import pyplot as plt

from skimage.exposure import rescale_intensity

<<<<<<< HEAD
nubes_sequence = [ # Fnalment no ho utilitzo
=======
nubes_sequence = [
>>>>>>> a60474462d71d79b45874a2a44ea096f812dd4ad
    # 2-10
    2, 3, 4, 5, 6, 7, 8, 9, 10,
    # 32-37
    32, 33, 34, 35, 36, 37,
    # 61-67S
    61, 62, 63, 64, 65, 66, 67,
    # 91-99
    91, 92, 93, 94, 95, 96, 97, 98, 99,
    # 121-128
    121, 122, 123, 124, 125, 126, 127, 128,
    # 151-159
    151, 152, 153, 154, 155, 156, 157, 158, 159,
    # 181-188
    181, 182, 183, 184, 185, 186, 187, 188,
    # 213-218
    213, 214, 215, 216, 217, 218,
    # 241-249
    241, 242, 243, 244, 245, 246, 247, 248, 249,
    # 272-278
    272, 273, 274, 275, 276, 277, 278,
    # 302-308
    302, 303, 304, 305, 306, 307, 308,
    # 333-340
    333, 334, 335, 336, 337, 338, 339, 340,
    # 363-368
    363, 364, 365, 366, 367, 368,
    # 393-398
    393, 394, 395, 396, 397, 398,
    # 425-432
    425, 426, 427, 428, 429, 430, 431, 432,
    # 454-462
    454, 455, 456, 457, 458, 459, 460, 461, 462,
    # 484-492
    484, 485, 486, 487, 488, 489, 490, 491, 492,
    # 515-560
    515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530,
    531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546,
    547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560,
    # 546-548 (already included in 515-560)
    546, 547, 548,
    # 608-617
    608, 609, 610, 611, 612, 613, 614, 615, 616, 617,
    # 647
    647
]

files = {
    'sentinel2april':  'datasets/S2_CAT_201804_UTM_WGS84_31N_10m.tif',
    'sentinel2august': 'datasets/S2_CAT_201808_UTM_WGS84_31N_10m.tif',
    '12classes': 'datasets/LC_2018_UTM_WGS84_31N_1m_12_Classes.tif',
    '12classesdownsampled':'datasets/LC_2018_UTM_WGS84_31N_1m_12_Classes_downsampled30x30_mode.tif',
}

def get_sentinel2_realcolor_image_old(image):

    image = image[[2, 1, 0]] / 1000.0
    #image_rescaled = rescale_intensity(image, in_range=(np.quantile(image, 0.01), np.quantile(image, 0.99)))
    image_rescaled = rescale_intensity(image, in_range=(np.quantile(image, 0.015), np.quantile(image, 0.98)))
    #image_rescaled = rescale_intensity(image, in_range=(0, 1))
    image_without_rescaling = image

    return image_rescaled, image_without_rescaling


def get_normalized(image):

    image_normalized = rescale_intensity(image, in_range=(np.quantile(image, 0.01), np.quantile(image, 0.99)))

    return image_normalized

def create_pngs_from_geotiff(sentinel_file, output_dir, subname):
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(sentinel_file) as src:
        img_width, img_height = src.width, src.height #dimensions of the sentinel-2 image
        tile_size = 1000
        count = 1
        
        # Loop through the entire image, creating non-overlapping 1000x1000 tiles
        for x in range(0, img_width, tile_size):
            for y in range(0, img_height, tile_size):
                # Define a window for the current tile
                window = Window(x, y, tile_size, tile_size)

                # Read the RGB bands 
                try:
                    image = src.read([1, 2, 3], window=window)
                    if image.size == 0:
                        print(f"Skipping empty tile at ({x}, {y})")
                        continue
                except Exception as e:
                    print(f"Error reading tile at ({x}, {y}): {e}")
                    continue

                filepath_normalized = os.path.join(output_dir, f"{subname}_tile_norm_{x}_{y}.png")
                filepath_not_normalized = os.path.join(output_dir, f"{subname}_tile_nonorm_{x}_{y}.png")

                # reordering channels and scaling pixel 
                image = image[[2, 1, 0]] / 1000.0

                # Normalization
                image_normalized = get_normalized(image)
                image_normalized = np.moveaxis(image_normalized, 0, -1)  
                plt.imsave(filepath_normalized, image_normalized)

                #Sense Clip els valors son superiors a 1 i no es pot gravar
                image_not_normalized = np.moveaxis(image, 0, -1)  # Adjust channel order for saving
                plt.imsave(filepath_not_normalized, np.clip(image_not_normalized, 0, 1)) 

                plt.close()

                count += 1
                print(f"Saved image {count}: for {x} - {y}")

                #if count == 100:
                    #break
    
    print(f"Total images saved: {count}")


def create_tiles_from_mask(mask_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with rasterio.open(mask_file) as src:
        img_width, img_height = src.width, src.height #dimensions of the sentinel-2 image
        tile_size = 1000
        count = 1
        
        # Loop through the entire image, creating non-overlapping 1000x1000 tiles
        for x in range(0, img_width, tile_size):
            for y in range(0, img_height, tile_size):
                # Define a window for the current tile
                window = Window(x, y, tile_size, tile_size)

                # Read the RGB bands 
                try:
                    image = src.read(1, window=window)
                    if image.size == 0:
                        print(f"Skipping empty tile at ({x}, {y})")
                        continue
                except Exception as e:
                    print(f"Error reading tile at ({x}, {y}): {e}")
                    continue

                filepath_mask = os.path.join(output_dir, f"mask_tile_{x}_{y}.png")
                plt.imsave(filepath_mask, image)
                plt.close()

                count += 1
                print(f"Saved image {count}: for {x} - {y}")

                #if count == 100:
                    #break
    
    print(f"Total images saved: {count} and expected 1800")

create_pngs_from_geotiff(files['sentinel2april'],  'D:\\tiles_training\\sentinel2april', 'april')
#create_pngs_from_geotiff(files['sentinel2august'], 'D:\\tiles_training\\sentinel2august', 'august')
#create_tiles_from_mask(files['12classesdownsampled'], 'D:\\tiles_training\\tiles_masks')