import json
import pandas as pd
import re
import requests
from tqdm import tqdm
import time
import numpy as np
import os

def geocode_subdistrict(subdistrict):
    """
    Geocode a subdistrict using the Open-Meteo Geocoding API.

    Parameters:
    - subdistrict (str): The name of the subdistrict to geocode.

    Returns:
    - tuple: (latitude, longitude) if found, else (np.nan, np.nan)
    """
    base_url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        'name': subdistrict,
        'count': 1,
        'language': 'en',
        'format': 'json'
    }

    try:
        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses
        data = response.json()

        if 'results' in data and len(data['results']) > 0:
            latitude = data['results'][0]['latitude']
            longitude = data['results'][0]['longitude']
            return latitude, longitude
        else:
            print(f"No geocoding result for subdistrict: {subdistrict}")
            return np.nan, np.nan
    except requests.exceptions.RequestException as e:
        print(f"Request error for subdistrict '{subdistrict}': {e}")
        return np.nan, np.nan

# Path to your JSON file
file_path = '/content/drive/MyDrive/wd/centaline/centa_final.json'

try:
    # Load the JSON file into a DataFrame
    centa_df = pd.read_json(file_path, lines=True)

    # Print the DataFrame columns
    print("Columns in the DataFrame:")
    print(centa_df.columns)

    # Initialize a list to collect all chartData entries
    all_chart_data = []

    # Iterate over each row in the DataFrame
    for index, row in centa_df.iterrows():
        chart_data_raw = row.get('ChartData', '')

        # Use regex to extract the 'chartData' array from the 'ChartData' string
        # This pattern captures everything between 'chartData:[' and the closing ']'
        match = re.search(r'chartData\s*:\s*(\[\{.*?\}\])\s*(?:,|;)', chart_data_raw, re.DOTALL)

        if match:
            chart_data_json_str = match.group(1)

            try:
                # Parse the extracted chartData array
                chart_data = json.loads(chart_data_json_str)

                # Iterate through each entry in the chartData array
                for entry in chart_data:
                    # Extract desired fields
                    extracted_entry = {
                        'District': row.get('District', None),
                        'Sub-District': row.get('Sub-District', None),
                        'Estate': row.get('Estate', None),
                        'Link': row.get('Link', None),
                        'edu_ter': entry.get('edu_ter', None),
                        'Type': entry.get('Type', None),
                        'Code': entry.get('Code', None),
                        'NameE': entry.get('NameE', None),
                        'DateValue': entry.get('DateValue', None),
                        'Period': entry.get('Period', None)
                    }
                    all_chart_data.append(extracted_entry)

            except json.JSONDecodeError as e:
                print(f"Error decoding chartData JSON for row {index}: {e}")
                continue  # Skip to the next row
        else:
            print(f"No 'chartData' found for row {index}.")
            continue  # Skip to the next row

    # Convert the list of extracted entries into a DataFrame
    chart_data_df = pd.DataFrame(all_chart_data)

    # Display the first few rows of the extracted chartData DataFrame
    print("\nExtracted ChartData:")
    print(chart_data_df.head())

    # Ensure the 'Sub-District' column exists
    if 'Sub-District' not in chart_data_df.columns:
        print("The 'Sub-District' column is missing in the extracted data.")
        unique_subdistricts = []
    else:
        # Extract unique subdistricts, excluding NaN values
        unique_subdistricts = chart_data_df['Sub-District'].dropna().unique()
        print(f"Number of unique subdistricts: {len(unique_subdistricts)}")

    # Define cache file path
    cache_file = '/content/drive/MyDrive/wd/centaline/subdistrict_geocoding_cache.csv'

    # Check if cache exists
    if os.path.exists(cache_file):
        geocode_cache_df = pd.read_csv(cache_file)
        geocode_cache = dict(zip(geocode_cache_df['Sub-District'], zip(geocode_cache_df['lat'], geocode_cache_df['lon'])))
        print(f"Loaded geocoding cache with {len(geocode_cache)} entries.")
    else:
        geocode_cache = {}
        print("No existing geocoding cache found. Starting fresh.")

    # Initialize a list to store geocoding results
    geocoding_results = []

    # Iterate over unique subdistricts with a progress bar
    for subdistrict in tqdm(unique_subdistricts, desc="Geocoding Subdistricts"):
        if subdistrict in geocode_cache:
            lat, lon = geocode_cache[subdistrict]
        else:
            lat, lon = geocode_subdistrict(subdistrict)
            geocode_cache[subdistrict] = (lat, lon)
            # To respect API rate limits, sleep for a short duration
            time.sleep(1)  # Adjust based on API's rate limit policy

        geocoding_results.append({
            'Sub-District': subdistrict,
            'lat': lat,
            'lon': lon
        })

    # Convert results to DataFrame
    geocoding_df = pd.DataFrame(geocoding_results)

    # Save the updated cache
    geocoding_df.to_csv(cache_file, index=False)
    print(f"Geocoding cache saved to {cache_file}.")

    # Merge the geocoding results with the original DataFrame
    chart_data_with_geo = chart_data_df.merge(geocoding_df, on='Sub-District', how='left')

    # Verify the merge
    print("\nDataFrame after merging geocoded data:")
    print(chart_data_with_geo.head())

    # Filter rows where 'Period' is 'Month' (case-insensitive)
    monthly = chart_data_with_geo[chart_data_with_geo['Period'].str.lower() == 'month']
    print(f"\nShape of monthly data: {monthly.shape}")

    # Filter rows where 'edu_ter' is not empty (not null)
    month_ne = monthly[monthly['edu_ter'].notna()].copy()
    print(f"Shape of month_ne data: {month_ne.shape}")

    # Add a column with the logarithm of 'edu_ter'
    month_ne['log'] = np.log(month_ne['edu_ter'])
    print("\nHead of month_ne DataFrame:")
    print(month_ne.head())

    # Optionally, save the extracted and geocoded data to a CSV file
    save_path = '/content/drive/MyDrive/wd/centaline/month_ne_with_lat_lon.csv'
    chart_data_with_geo.to_csv(save_path, index=False)
    print(f"\nExtracted and geocoded data saved to: {save_path}")

except FileNotFoundError:
    print(f"File not found at path: {file_path}")
except ValueError as e:
    print(f"Error reading JSON file: {e}")
