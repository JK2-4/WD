
# Install necessary libraries
#!pip install --upgrade libpysal geopandas matplotlib seaborn esda contextily spreg
import libpysal
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from libpysal.weights import Queen
import spreg  # Updated import for spatial regression
import contextily as ctx
import warnings
warnings.filterwarnings('ignore')

# Define the shapefile path
shp_path = '/content/drive/MyDrive/wd/summary rent/Summary_Statistics_of_Government_Rent_Roll_by_district_in_Hong_Kong_1-polygon.shp'

# Load the shapefile using geopandas
gdf = gpd.read_file(shp_path)

# Inspect the data
print(gdf.head())
print(gdf.crs)

# Reproject to Web Mercator for compatibility with contextily
gdf = gdf.to_crs(epsg=3857)

# Create Queen contiguity spatial weights matrix
w = Queen.from_dataframe(gdf)

# Row-standardize the weights
w.transform = 'R'

# Verify normalization - Calculate and print row sums for each observation to confirm normalization
row_sums = [sum(w[row].values()) for row in w.neighbors]
print("Row sums (should be ~1 for all rows):", row_sums)

# Visualization: Choropleth Map
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
gdf.plot(column='Not_Exceed', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
plt.title('Government Rent Values by District in Hong Kong', fontsize=15)
plt.axis('off')
plt.show()

# Visualization: Spatial Weights (Neighbor Links)
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
gdf.plot(ax=ax, color='white', edgecolor='black')

for district, neighbors in w.neighbors.items():
    for neighbor in neighbors:
        district_geom = gdf.geometry.iloc[district]
        neighbor_geom = gdf.geometry.iloc[neighbor]
        district_centroid = district_geom.centroid
        neighbor_centroid = neighbor_geom.centroid
        xs = [district_centroid.x, neighbor_centroid.x]
        ys = [district_centroid.y, neighbor_centroid.y]
        ax.plot(xs, ys, color='blue', linewidth=0.5, alpha=0.5)

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
plt.title('Spatial Weights (Neighbor Links) Between Districts in Hong Kong', fontsize=15)
plt.axis('off')
plt.show()


# Import libraries
import libpysal
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from libpysal.weights import Queen
import spreg  # Updated import for spatial regression
import contextily as ctx
import warnings
warnings.filterwarnings('ignore')

# Define the shapefile path
shp_path = '/content/drive/MyDrive/wd/summary rent/Summary_Statistics_of_Government_Rent_Roll_by_district_in_Hong_Kong_1-polygon.shp'

# Load the shapefile using geopandas
gdf = gpd.read_file(shp_path)

# Reproject to Web Mercator for compatibility with contextily
gdf = gdf.to_crs(epsg=3857)

# Create Queen contiguity spatial weights matrix
w = Queen.from_dataframe(gdf)

# Row-standardize the weights
w.transform = 'R'

# Save Spatial Weights Matrix to CSV
weights_list = []
for row in w.neighbors:
    neighbors = w.neighbors[row]
    for neighbor in neighbors:
        weights_list.append([row, neighbor, w[row][neighbor]])

weights_df = pd.DataFrame(weights_list, columns=['District_ID', 'Neighbor_ID', 'Weight'])
weights_csv_path = '/content/drive/MyDrive/wd/summary rent/spatial_weights_matrix.csv'
weights_df.to_csv(weights_csv_path, index=False)
print(f"Spatial weights matrix saved to: {weights_csv_path}")

# Save GeoDataFrame (without geometry) to CSV
gdf_no_geometry = gdf.drop(columns='geometry')
gdf_csv_path = '/content/drive/MyDrive/wd/summary rent/gdf_data.csv'
gdf_no_geometry.to_csv(gdf_csv_path, index=False)
print(f"GeoDataFrame data saved to: {gdf_csv_path}")
