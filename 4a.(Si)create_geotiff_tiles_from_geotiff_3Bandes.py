from osgeo import gdal, osr
import os

# Define the input and output paths
input_file = "datasets/12classes_downsampled_to_30k_30k.tif"
output_dir = "D:\\tiles_training\\tiles_masks\\tiffs"

tile_size = 1000  # Size of each tile

# Open the source GeoTIFF
src_ds = gdal.Open(input_file)
if not src_ds:
    raise FileNotFoundError(f"Could not open file {input_file}")

# Get the georeferencing info
geo_transform = src_ds.GetGeoTransform()
projection = src_ds.GetProjection()
band = src_ds.GetRasterBand(1)
nodata_value = band.GetNoDataValue()

# Get dimensions of the source image
x_size = src_ds.RasterXSize
y_size = src_ds.RasterYSize

# Calculate the number of tiles in x and y directions
x_tiles = (x_size + tile_size - 1) // tile_size
y_tiles = (y_size + tile_size - 1) // tile_size

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Loop through each tile
for i in range(x_tiles):
    for j in range(y_tiles):
        try:
            x_offset = i * tile_size
            y_offset = j * tile_size
            x_tile_size = min(tile_size, x_size - x_offset)
            y_tile_size = min(tile_size, y_size - y_offset)

            output_file = os.path.join(output_dir, f"tile_{i}_{j}.tif")
            driver = gdal.GetDriverByName("GTiff")
            dst_ds = driver.Create(output_file, x_tile_size, y_tile_size, 3, gdal.GDT_Byte)
            if not dst_ds:
                raise RuntimeError(f"Could not create output file {output_file}")

            tile_geo_transform = (
                geo_transform[0] + x_offset * geo_transform[1],
                geo_transform[1],
                geo_transform[2],
                geo_transform[3] + y_offset * geo_transform[5],
                geo_transform[4],
                geo_transform[5],
            )
            dst_ds.SetGeoTransform(tile_geo_transform)
            dst_ds.SetProjection(projection)

            # Read and scale RGB bands
            all_bands = []

            for target_band in [4, 3, 2]:  # Sentinel-2 RGB bands
                band = src_ds.GetRasterBand(target_band)
                data = band.ReadAsArray(x_offset, y_offset, x_tile_size, y_tile_size)
                all_bands.append(data)

            # Calculate global min/max for consistent scaling
            global_min = min([band.min() for band in all_bands])
            global_max = max([band.max() for band in all_bands])

            # Normalize each band to 0–255 using the global range
            for band_index, data in enumerate(all_bands, start=1):
                if global_max > global_min:
                    data = ((data - global_min) / (global_max - global_min) * 255).astype("uint8")
                dst_band = dst_ds.GetRasterBand(band_index)
                dst_band.WriteArray(data)
                nodata_value = src_ds.GetRasterBand(4).GetNoDataValue()  # Assume all bands share NoData
                if nodata_value is not None:
                    dst_band.SetNoDataValue(nodata_value)

            dst_ds.FlushCache()
            dst_ds = None

            print(f"Tile {i}, {j} written to {output_file}")
        except Exception as e:
            print(f"Error processing tile {i}, {j}: {e}")

