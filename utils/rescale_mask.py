import os
from osgeo import gdal

def resample_mask(input_mask_file, output_mask_file, target_size=(30000, 30000)):
    # Abrir el archivo original de la máscara con GDAL
    dataset = gdal.Open(input_mask_file)
    
    if dataset is None:
        print(f"Error: no se pudo abrir el archivo {input_mask_file}")
        return

    # Obtener las dimensiones de la máscara original
    original_width = dataset.RasterXSize
    original_height = dataset.RasterYSize

    print(f"Original size: {original_width} x {original_height}")

    # Configurar la transformación para remuestreo
    options = gdal.WarpOptions(
        width=target_size[0],
        height=target_size[1],
        resampleAlg='near'  # Nearest Neighbor
    )

    # Realizar el remuestreo
    gdal.Warp(output_mask_file, dataset, options=options)

    # Copiar la paleta de colores del archivo original al archivo reescalado
    original_band = dataset.GetRasterBand(1)  # Asumiendo que la paleta está en la banda 1
    output_dataset = gdal.Open(output_mask_file, gdal.GA_Update)

    if output_dataset:
        output_band = output_dataset.GetRasterBand(1)
        color_table = original_band.GetColorTable()
        if color_table:
            output_band.SetColorTable(color_table)
            print("Paleta de colores transferida al archivo reescalado.")
        else:
            print("El archivo original no tiene una paleta de colores.")

        # Cerrar el archivo reescalado
        output_dataset = None

    # Cerrar el dataset original
    dataset = None

    print(f"Archivo de máscara reescalado guardado en {output_mask_file}")

# Ejemplo de uso
input_mask_file = 'D:\data\modeloData\labels\LC_2018_UTM_WGS84_31N_1m_12_Classes.tif'  # Archivo de la máscara original
output_mask_file = 'D:\data\modeloData\labels\escale12clases.tif'  # Archivo de la máscara reescalada

resample_mask(input_mask_file, output_mask_file, target_size=(30000, 30000))