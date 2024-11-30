import geopandas as gpd
import pandas as pd
import folium

# Load the GeoJSON file
geojson_file = 'https://hub.arcgis.com/api/v3/datasets/103c740e57424f9882c5303c333487f1_0/downloads/data?format=geojson&spatialRefId=4326&where=1%3D1'
gdf = gpd.read_file(geojson_file)

# Display the first few rows
print(gdf.head())

# Convert relevant columns to numeric values
numeric_columns = [
    "Not_Exceeding_Min_RV_Number",
    "Not_Exceeding_Min_RV_Number_TC",
    "Above_Min_RV_Number",
    "Above_Min_RV_Number_TC",
    "Above_Min_RV_HKD_In_Thousand",
    "Above_Min_RV_HKD_In_Thousand_TC",
    "SHAPE__Area",
    "SHAPE__Length"
]

for col in numeric_columns:
    gdf[col] = pd.to_numeric(gdf[col], errors='coerce')

# Drop rows with missing values
gdf_clean = gdf.dropna(subset=numeric_columns + ['District'])

# Convert LastUpdate to datetime, then to a string for JSON serialization
gdf_clean['LastUpdate'] = pd.to_datetime(gdf_clean['LastUpdate'], format='%Y-%m', errors='coerce')
gdf_clean['LastUpdate_str'] = gdf_clean['LastUpdate'].dt.strftime('%Y-%m')

# Drop the original datetime column to avoid JSON serialization issues
gdf_clean = gdf_clean.drop(columns=['LastUpdate'])

# Verify the changes
print(gdf_clean[['District', 'Above_Min_RV_HKD_In_Thousand', 'LastUpdate_str']].head())

# Initialize a Folium map centered around Hong Kong
m = folium.Map(location=[22.3193, 114.1694], zoom_start=10)

# Add GeoJSON to the map with color-coded rent value information
folium.GeoJson(
    gdf_clean,
    name='geojson',
    style_function=lambda feature: {
        'fillColor': '#3186cc' if feature['properties']['Above_Min_RV_HKD_In_Thousand'] > 100000 else '#ffcccb',
        'color': 'black',
        'weight': 1,
        'fillOpacity': 0.5,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=['District', 'Above_Min_RV_HKD_In_Thousand', 'LastUpdate_str'],  # Use LastUpdate_str instead of LastUpdate
        aliases=['District', 'Rent Value (HKD Thousands)', 'Last Updated'],
        localize=True
    )
).add_to(m)

# Display the map
m
