import os
import requests
import zipfile
import tempfile
from glob import glob
import pandas as pd
import numpy as np
from osgeo import gdal, osr
from tqdm import tqdm
from pathlib import Path

def aggregate_raster_values(raster_data, no_data, px, py, raster_x_size, raster_y_size, window_size=5):
    """
    Aggregate raster values within a square window around the given pixel coordinates.

    Parameters:
    - raster_data: 2D NumPy array of raster values.
    - no_data: The no-data value for the raster.
    - px, py: Central pixel coordinates.
    - raster_x_size, raster_y_size: Dimensions of the raster.
    - window_size: The size of the window (must be odd to have a central pixel).

    Returns:
    - The aggregated (average) raster value or np.nan if no valid data is found.
    """
    half_window = window_size // 2
    # Define window boundaries
    x_min = max(px - half_window, 0)
    x_max = min(px + half_window + 1, raster_x_size)
    y_min = max(py - half_window, 0)
    y_max = min(py + half_window + 1, raster_y_size)

    window = raster_data[y_min:y_max, x_min:x_max]

    # Mask no-data values
    if no_data is not None:
        valid_mask = window != no_data
        valid_data = window[valid_mask]
    else:
        valid_data = window.flatten()

    if valid_data.size > 0:
        return np.mean(valid_data)
    else:
        return np.nan

def process_tiff(tiff_path, df, window_size=5):
    """
    Processes a single TIFF file: aggregates raster values around each point in the DataFrame.

    Parameters:
    - tiff_path: Path to the TIFF file.
    - df: Pandas DataFrame containing 'latitude' and 'longitude' columns.
    - window_size: The size of the aggregation window.

    Returns:
    - A Pandas Series with aggregated values for each point.
    - A string representing the new column name based on the TIFF filename.
    """
    # Open the raster
    ds = gdal.Open(tiff_path)
    if ds is None:
        print(f"Could not open {tiff_path}")
        return pd.Series([np.nan]*len(df)), None

    band = ds.GetRasterBand(1)
    raster_data = band.ReadAsArray()
    no_data = band.GetNoDataValue()
    geotrans = ds.GetGeoTransform()
    proj = ds.GetProjection()
    raster_x_size = ds.RasterXSize
    raster_y_size = ds.RasterYSize

    # Define spatial references
    csv_srs = osr.SpatialReference()
    csv_srs.SetWellKnownGeogCS("WGS84")  # Assuming CSV coordinates are in WGS84

    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(proj)
    transform = osr.CoordinateTransformation(csv_srs, raster_srs)

    # Extract geotransform parameters
    origin_x = geotrans[0]
    origin_y = geotrans[3]
    pixel_width = geotrans[1]
    pixel_height = geotrans[5]

    # Determine the new column name based on the TIFF filename
    tiff_filename = os.path.basename(tiff_path)
    column_name = os.path.splitext(tiff_filename)[0].replace(' ', '_').replace('-', '_')

    # Initialize a list to store aggregated values
    aggregated_values = []

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {tiff_filename}"):
        lat = row['latitude']
        lon = row['longitude']

        # Transform coordinates
        try:
            x, y, _ = transform.TransformPoint(lon, lat)
        except Exception as e:
            print(f"Error transforming point ({lon}, {lat}): {e}")
            aggregated_values.append(np.nan)
            continue

        # Convert geographic coordinates to pixel coordinates
        px = int((x - origin_x) / pixel_width)
        py = int((origin_y - y) / abs(pixel_height))

        # Check if pixel coordinates are within raster bounds
        if 0 <= px < raster_x_size and 0 <= py < raster_y_size:
            aggregated = aggregate_raster_values(
                raster_data, no_data, px, py,
                raster_x_size, raster_y_size,
                window_size=window_size
            )
            aggregated_values.append(aggregated)
        else:
            # If out of bounds, assign NaN
            aggregated_values.append(np.nan)

    # Create a Pandas Series for the new column
    aggregated_series = pd.Series(aggregated_values, index=df.index)

    return aggregated_series, column_name

def main(): 
  tiff = '/content/drive/MyDrive/wd/summary_rent/tif_data/lcz.tif'
  csv_path = '/content/drive/MyDrive/wd/massive data/lat lon stations.csv'
  output_csv = '/content/drive/MyDrive/wd/massive data/lat lon stations.csv'
  window_size = 5  # Example: 5 for a 5x5 window
  print("Reading the CSV file...")
  df = pd.read_csv(csv_path)
  # Ensure the CSV has 'latitude' and 'longitude' columns
  if not {'latitude', 'longitude'}.issubset(df.columns):
      raise ValueError("CSV must contain 'latitude' and 'longitude' columns.")
      
  aggregated_series, column_name = process_tiff(tiff, df, window_size=window_size)

  if column_name:
      # Add the new column to the DataFrame
      df[column_name] = aggregated_series
      print(f"Added column '{column_name}' to the DataFrame.")
  else:
      print(f"Skipped adding column for {tiff} due to processing issues.")

  # Identify columns that were added (assuming they were named based on TIFF filenames)
  aggregated_columns = [col for col in df.columns if col not in ['latitude', 'longitude']]

  missing_values = df[aggregated_columns].isnull().sum()

  if missing_values.any():
      print("\nWarning: Some coordinates did not find valid raster values within the aggregation window.")
      print("Consider increasing the window size or verifying the coordinate accuracy.\n")
      print(missing_values[missing_values > 0])
  else:
      print("\nAll coordinates have valid raster values.")

  print(f"\nSaving the updated CSV to {output_csv}...")
  df.to_csv(output_csv, index=False)
  print("Process completed successfully!")

if __name__ == "__main__":
    main()
