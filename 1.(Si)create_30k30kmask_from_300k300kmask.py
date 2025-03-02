"""
Autor: Esteve Graells
Data: 2 de març de 2025

Utilitzo aquest script per a reduir la resolució d'un fitxer GeoTIFF gran utilitzant la remostreig de veí més proper.
El procés de reducció de resolució redueix la resolució del GeoTIFF d'entrada per un factor d'escala especificat,
preservant els valors de classe discrets sense interpolació. El resultat és un fitxer GeoTIFF amb resolució reduïda.

Paràmetres:
    input_tiff (str): Ruta al fitxer GeoTIFF d'entrada.
    output_tiff (str): Ruta al fitxer GeoTIFF amb resolució reduïda.
    scale_factor (int): Factor pel qual es redueix la resolució (per defecte: reducció de 10x).
"""

import os
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import  Resampling
import numpy as np


def downsample_geotiff_v1(input_tiff, output_tiff, scale_factor=10):
    """
    Downsamples a large GeoTIFF file using nearest neighbor resampling.
    
    :param input_tiff: Path to the input GeoTIFF file.
    :param output_tiff: Path to the output downsampled GeoTIFF file.
    :param scale_factor: Factor by which to downsample (default: 10x reduction).
    """
    with rasterio.open(input_tiff) as src:
        new_width = src.width // scale_factor
        new_height = src.height // scale_factor
        
        profile = src.profile.copy()
        # Update the metadata to reflect the new downsampled dimensions and transformation
        profile.update({
            "width": new_width,
            "height": new_height,
            "transform": src.transform * src.transform.scale(scale_factor, scale_factor)
        })
        
        with rasterio.open(output_tiff, "w", **profile) as dst:
            # Using nearest neighbor resampling to preserve discrete class values
            # This ensures that no interpolation occurs, preventing unintended class mixing
            data = src.read(
                1, ## Read the first band
                out_shape=(new_height, new_width),
                resampling=Resampling.mode
            ) 
            dst.write(data, 1)
    
    print(f"Downsampling completed: {output_tiff}")

if __name__ == "__main__":
    input_tiff =  r"D:/aidl_projecte/datasets/LC_2018_UTM_WGS84_31N_1m_12_Classes.tif"
    output_tiff = r"D:/aidl_projecte/datasets/LC_2018_UTM_WGS84_31N_1m_12_Classes_downsampled30x30_mode.tif"
    
    downsample_geotiff_v1(input_tiff, output_tiff)


