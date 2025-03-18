import os
from osgeo import gdal

''' 
This script resizes a raster mask using GDAL, reducing its dimensions from 300,000 to 30,000 pixels. 
It performs nearest-neighbor resampling and preserves the color table if present. 
'''

def resample_mask(input_mask_file, output_mask_file, target_size=(30000, 30000)):
    dataset = gdal.Open(input_mask_file)
    
    if dataset is None:
        print(f"Error: we can't open the file: {input_mask_file}")
        return

    original_width = dataset.RasterXSize
    original_height = dataset.RasterYSize

    print(f"Original size: {original_width} x {original_height}")

    options = gdal.WarpOptions(
        width=target_size[0],
        height=target_size[1],
        resampleAlg='near'
    )

    gdal.Warp(output_mask_file, dataset, options=options)

    original_band = dataset.GetRasterBand(1) 
    output_dataset = gdal.Open(output_mask_file, gdal.GA_Update)

    if output_dataset:
        output_band = output_dataset.GetRasterBand(1)
        color_table = original_band.GetColorTable()
        if color_table:
            output_band.SetColorTable(color_table)
            print("Color palette transferred to the rescaled file.")
        else:
            print("The original file does not have a color palette.")

        output_dataset = None

    dataset = None

    print(f"Rescaled mask file saved at {output_mask_file}")

input_mask_file = './datasets/LC_2018_UTM_WGS84_31N_1m_12_Classes.tif'  # Original mask
output_mask_file = './rescale_mask/mask.tif'  # Rescaled mask

resample_mask(input_mask_file, output_mask_file, target_size=(30000, 30000))