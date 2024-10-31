import osmium
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import os
import matplotlib.pyplot as plt
import string

# Define the folder paths
input_folder = '/content/drive/MyDrive/wd/osmosis-0.49.2/osmo/output_osm_pbf'
output_folder = '/content/drive/MyDrive/wd/osmosis-0.49.2/osmo/output2/'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

class OSMHandler(osmium.SimpleHandler):
    def __init__(self):
        super(OSMHandler, self).__init__()
        self.nodes = {}
        self.ways = []

    def node(self, n):
        # Store node ID with its longitude and latitude
        self.nodes[n.id] = (n.location.lon, n.location.lat)

    def way(self, w):
        # Store way ID, list of node IDs, and tags
        self.ways.append({
            'id': w.id,
            'nodes': [node.ref for node in w.nodes],
            'tags': {tag.k: tag.v for tag in w.tags}
        })

# Loop through all .osm.pbf files in the folder
for file_name in os.listdir(input_folder):
    if file_name.endswith('.osm.pbf'):
        # Define the file path
        pbf_path = os.path.join(input_folder, file_name)
        
        # Extract district name from the file name (e.g., 'ISLANDS' from 'ISLANDS.osm.pbf')
        district_name = file_name.split('.')[0]

        # Initialize handler
        handler = OSMHandler()

        # Apply handler to the .osm.pbf file
        handler.apply_file(pbf_path, locations=True)

        # Extracted Data
        nodes_data = handler.nodes  # Dictionary: {node_id: (lon, lat)}
        ways_data = handler.ways    # List of ways with 'id', 'nodes', 'tags'

        # Convert ways_data to DataFrame
        ways_df = pd.DataFrame(ways_data)

        # For nodes, create a DataFrame from the dictionary
        nodes_df = pd.DataFrame.from_dict(handler.nodes, orient='index', columns=['lon', 'lat'])
        nodes_df.reset_index(inplace=True)
        nodes_df.rename(columns={'index': 'id'}, inplace=True)

        # Create a mapping from node IDs to their coordinates
        node_id_to_coords = nodes_df.set_index('id')[['lon', 'lat']].to_dict('index')

        # Function to reconstruct geometry for a way
        def reconstruct_geometry(nodes_list, node_mapping):
            try:
                coords = [(node_mapping[node_id]['lon'], node_mapping[node_id]['lat']) for node_id in nodes_list]
                if coords[0] == coords[-1]:
                    return Polygon(coords)
                else:
                    return LineString(coords)
            except KeyError as e:
                print(f"Missing node ID: {e}")
                return None

        # Apply the function to reconstruct geometries
        ways_df['geometry'] = ways_df['nodes'].apply(lambda x: reconstruct_geometry(x, node_id_to_coords))

        # Drop ways where geometry couldn't be reconstructed
        ways_df = ways_df.dropna(subset=['geometry'])

        # Create GeoDataFrame for nodes
        gdf_nodes = gpd.GeoDataFrame(
            nodes_df,
            geometry=gpd.points_from_xy(nodes_df.lon, nodes_df.lat),
            crs="EPSG:4326"  # WGS84 Latitude/Longitude
        )

        # Create GeoDataFrame for ways with reconstructed geometries
        gdf_ways = gpd.GeoDataFrame(
            ways_df[['id', 'tags', 'geometry']],
            geometry='geometry',
            crs="EPSG:4326"  # WGS84 Latitude/Longitude
        )

        # Combine nodes and ways into a single GeoDataFrame
        gdf_combined = pd.concat([gdf_nodes, gdf_ways], ignore_index=True)

        # Project to Web Mercator (meters)
        gdf_combined = gdf_combined.to_crs(epsg=3857)

        # Define buffer diameters in meters
        diameters = [7000, 12000]  # in meters

        # Calculate the centroid of all geometries in the district
        centroid = gdf_combined.geometry.unary_union.centroid

        # Create buffer geometries
        buffers = {}
        for diameter in diameters:
            radius = diameter / 2  # Buffer radius in meters
            buffer_geom = centroid.buffer(radius)
            buffers[diameter] = buffer_geom
        

        def calculate_land_surface(gdf, buffer_geom):
            print("Calculating land surface characteristics.")
            # Clip data to buffer
            clipped = gdf[gdf.intersects(buffer_geom)].copy()
            print(f"Records within buffer: {len(clipped)}")


            # Filter only Polygon and MultiPolygon geometries
            clipped = clipped[clipped.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
            print(f"Polygon records within buffer: {len(clipped)}")


            # Initialize area calculations
            total_area = buffer_geom.area  # In square meters

            # Define land use categories based on tags
            built_up = clipped[clipped['tags'].apply(lambda x: isinstance(x, dict) and ('building' in x or 'highway' in x))].copy()
            vegetated = clipped[clipped['tags'].apply(lambda x: isinstance(x, dict) and (x.get('natural') == 'vegetation' or x.get('landuse') in ['forest', 'grass', 'orchard']))].copy()
            bare_soil = clipped[clipped['tags'].apply(lambda x: isinstance(x, dict) and x.get('landuse') == 'bare_soil')].copy()
            rock = clipped[clipped['tags'].apply(lambda x: isinstance(x, dict) and x.get('natural') == 'rock')].copy()
            water = clipped[clipped['tags'].apply(lambda x: isinstance(x, dict) and (x.get('natural') == 'water' or x.get('waterway') is not None))].copy()

            built_up = gpd.GeoDataFrame(built_up, geometry='geometry', crs=gdf.crs)
            vegetated = gpd.GeoDataFrame(vegetated, geometry='geometry', crs=gdf.crs)
            bare_soil = gpd.GeoDataFrame(bare_soil, geometry='geometry', crs=gdf.crs)
            rock = gpd.GeoDataFrame(rock, geometry='geometry', crs=gdf.crs)
            water = gpd.GeoDataFrame(water, geometry='geometry', crs=gdf.crs)


            # Calculate areas
            built_up_area = built_up.geometry.area.sum()
            vegetated_area = vegetated.geometry.area.sum()
            bare_soil_area = bare_soil.geometry.area.sum()
            rock_area = rock.geometry.area.sum()
            water_area = water.geometry.area.sum()

            print(f"Built-up area: {built_up_area} m²")
            print(f"Vegetated area: {vegetated_area} m²")
            print(f"Bare soil area: {bare_soil_area} m²")
            print(f"Rock area: {rock_area} m²")
            print(f"Water area: {water_area} m²")

            # Calculate percentages
            built_up_pct = (built_up_area / total_area) * 100 if total_area > 0 else 0
            vegetated_pct = (vegetated_area / total_area) * 100 if total_area > 0 else 0
            bare_soil_pct = (bare_soil_area / total_area) * 100 if total_area > 0 else 0
            rock_pct = (rock_area / total_area) * 100 if total_area > 0 else 0
            water_pct = (water_area / total_area) * 100 if total_area > 0 else 0

            return {
                '% built-up': built_up_pct,
                '% vegetated': vegetated_pct,
                '% bare soil': bare_soil_pct,
                '% rock': rock_pct,
                '% water': water_pct
            }

        def calculate_vegetation(gdf, buffer_geom):
            print("Calculating vegetation characteristics.")
            # Clip data to buffer
            clipped = gdf[gdf.intersects(buffer_geom)].copy()
            print(f"Records within buffer for vegetation: {len(clipped)}")


            # Filter only Polygon and MultiPolygon geometries
            clipped = clipped[clipped.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
            print(f"Polygon records within buffer: {len(clipped)}")

            vegetated = clipped[clipped['tags'].apply(lambda x: isinstance(x, dict) and (x.get('natural') == 'vegetation' or x.get('landuse') in ['forest', 'grass', 'orchard']))]
            print(f"Vegetated records: {len(vegetated)}")

            dense_veg = vegetated[vegetated['tags'].apply(lambda x: isinstance(x, dict) and x.get('density') == 'dense')].copy()
            sparse_veg = vegetated[vegetated['tags'].apply(lambda x: isinstance(x, dict) and x.get('density') == 'sparse')].copy()

            # Ensure they remain GeoDataFrames by explicitly setting their geometry
            dense_veg = gpd.GeoDataFrame(dense_veg, geometry='geometry', crs=gdf.crs)
            sparse_veg = gpd.GeoDataFrame(sparse_veg, geometry='geometry', crs=gdf.crs)

            print(f"Dense vegetation records: {len(dense_veg)}")
            print(f"Sparse vegetation records: {len(sparse_veg)}")

            total_area = buffer_geom.area

            dense_area = dense_veg.geometry.area.sum()
            sparse_area = sparse_veg.geometry.area.sum()

            print(f"Dense vegetation area: {dense_area} m²")
            print(f"Sparse vegetation area: {sparse_area} m²")

            dense_pct = (dense_area / total_area) * 100 if total_area > 0 else 0
            sparse_pct = (sparse_area / total_area) * 100 if total_area > 0 else 0

            return {
                '% dense/high vegetation': dense_pct,
                '% sparse/low vegetation': sparse_pct
            }

        def calculate_urban_geometry(gdf, buffer_geom):
            print("Calculating urban geometry characteristics.")
            # Clip data to buffer
            clipped = gdf[gdf.intersects(buffer_geom)].copy()
            print(f"Records within buffer for urban geometry: {len(clipped)}")


            # Filter only Polygon and MultiPolygon geometries
            clipped = clipped[clipped.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
            print(f"Polygon records within buffer: {len(clipped)}")

            buildings = clipped[clipped['tags'].apply(lambda x: isinstance(x, dict) and x.get('building') is not None)].copy()
            roads = clipped[clipped['tags'].apply(lambda x: isinstance(x, dict) and x.get('highway') is not None)].copy()


            buildings = gpd.GeoDataFrame(buildings, geometry='geometry', crs=gdf.crs)
            roads = gpd.GeoDataFrame(roads, geometry='geometry', crs=gdf.crs)

            print(f"Buildings within buffer: {len(buildings)}")
            print(f"Roads within buffer: {len(roads)}")

            total_area = buffer_geom.area

            building_area = buildings.geometry.area.sum()
            print(f"Total building area: {building_area} m²")

            # Function to get road width
            def get_road_width(tags):
                if 'width' in tags:
                    try:
                        return float(tags['width'])
                    except:
                        return 10  # default width
                else:
                    return 10  # default width

            roads['road_width'] = roads['tags'].apply(lambda x: get_road_width(x))
            road_area = (roads.geometry.length * roads['road_width']).sum()
            print(f"Total road area: {road_area} m²")

            building_fraction = (building_area / total_area) * 100 if total_area > 0 else 0
            road_fraction = (road_area / total_area) * 100 if total_area > 0 else 0

            print(f"Building Fraction: {building_fraction:.2f}%")
            print(f"Road/Impervious Cover Fraction: {road_fraction:.2f}%")

            # SVF (Sky View Factor) estimation
            average_svf = 1 - (building_fraction / 100)  # Simplistic estimation
            average_svf = max(0, min(average_svf, 1))  # Ensure SVF is between 0 and 1
            print(f"Average SVF: {average_svf:.2f}")

            # Function to get building height
            def get_building_height(tags):
                if 'height' in tags:
                    try:
                        return float(tags['height'])
                    except:
                        pass
                if 'building:levels' in tags:
                    try:
                        return float(tags['building:levels']) * 3  # Approximate height per level
                    except:
                        pass
                return pd.NA  # Use NaN for missing values

            buildings['building_height'] = buildings['tags'].apply(lambda x: get_building_height(x))
            building_heights = buildings['building_height'].dropna()
            print(f"Number of buildings with height data: {len(building_heights)}")

            average_height = building_heights.mean() if not building_heights.empty else 0
            std_height = building_heights.std() if not building_heights.empty else 0
            print(f"Average Building Height: {average_height:.2f} m")
            print(f"Std Dev of Building Height: {std_height:.2f} m")

            return {
                'Building fraction (%)': building_fraction,
                'Road/impervious cover fraction (%)': road_fraction,
                'Average SVF': average_svf,
                'Average building height (m)': average_height,
                'Std dev of building height (m)': std_height
            }

        def calculate_metrics(gdf, buffers):
            metrics = {}
            for diameter, buffer_geom in buffers.items():
                # Calculate metrics (land surface, vegetation, urban geometry)
                land_surface = calculate_land_surface(gdf, buffer_geom)
                vegetation = calculate_vegetation(gdf, buffer_geom)
                urban_geometry = calculate_urban_geometry(gdf, buffer_geom)
                # Combine all metrics into a single dictionary
                metrics[diameter] = {**land_surface, **vegetation, **urban_geometry}
            return metrics

        # Calculate metrics for all buffer zones
        all_metrics = calculate_metrics(gdf_combined, buffers)

        # Convert metrics dictionary to DataFrame
        metrics_df = pd.DataFrame(all_metrics).T  # Transpose to have diameters as rows
        metrics_df.index.name = 'Buffer_Diameter_m'
        metrics_df = metrics_df.reset_index()

        # Define the output path for the metrics CSV
        output_csv_path = os.path.join(output_folder, f'{district_name}_results.csv')

        # Save the metrics to CSV
        metrics_df.to_csv(output_csv_path, index=False)

        print(f"\nMetrics saved to {output_csv_path}")

        # Visualization for each buffer
        def visualize_buffer(gdf, buffer_geom, diameter, district_name):
            fig, ax = plt.subplots(figsize=(10, 10))
            gdf.plot(ax=ax, color='blue', alpha=0.5, markersize=10, label='OSM Elements')
            gpd.GeoSeries([buffer_geom]).boundary.plot(ax=ax, color='red', label=f'{diameter}m Buffer')
            plt.title(f'Buffer {diameter}m around Centroid: {district_name} district')
            plt.legend()
            plt.xlabel('Easting (meters)')
            plt.ylabel('Northing (meters)')
            output_plot_path = os.path.join(output_folder, f'{district_name}_{diameter}m_buffer.png')
            plt.savefig(output_plot_path)
            plt.close()
            print(f"Visualization saved to {output_plot_path}")

        # Save visualizations for each buffer
        for diameter, buffer_geom in buffers.items():
            visualize_buffer(gdf_combined, buffer_geom, diameter, district_name)



######################################### FINAL COMPILATION 

# !pip install pillow

import pandas as pd
import glob
import os

# Path to your directory containing CSV files
path = "/content/drive/MyDrive/wd/osmosis-0.49.2/osmo/output2"
all_files = glob.glob(os.path.join(path, "*.csv"))

# Initialize an empty list to hold each file's DataFrame
dataframes = []

# Loop through each file, read it, and add a 'District' column
for file in all_files:
    # Extract the base name of the file without extension for the District column
    district_name = os.path.basename(file).replace("_results.csv", "")
    
    # Read the file into a DataFrame
    df = pd.read_csv(file)
    
    # Add a new column 'District' with the district name for each row
    df['District'] = district_name
    
    # Append the DataFrame to the list
    dataframes.append(df)

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv("/content/drive/MyDrive/wd/osmosis-0.49.2/osmo/compiled_output.csv", index=False)
print("Compiled CSV saved successfully with 'District' column.")



from PIL import Image
import glob
import os
import math

# Define path to your PNG files
path = "/content/drive/MyDrive/wd/osmosis-0.49.2/osmo/output2/*.png"
output_path = "/content/drive/MyDrive/wd/osmosis-0.49.2/osmo/combined_output.png"

# Load all PNG files
image_files = glob.glob(path)

# Sort images to maintain order if required
image_files.sort()

# Load all images
images = [Image.open(img) for img in image_files]

# Determine grid size for 37 images (e.g., 7x6 grid)
grid_size = (7, 6) if len(images) == 37 else (math.ceil(math.sqrt(len(images))),) * 2

# Get single image dimensions
width, height = images[0].size

# Calculate size for the final image
grid_width = grid_size[0] * width
grid_height = grid_size[1] * height

# Create an empty image with calculated grid dimensions
combined_image = Image.new('RGB', (grid_width, grid_height))

# Paste images into the grid
for index, img in enumerate(images):
    row = index // grid_size[0]
    col = index % grid_size[0]
    combined_image.paste(img, (col * width, row * height))

# Save the final combined image
combined_image.save(output_path, "PNG")
print("Combined image saved at:", output_path)

##################################################### if issues - 

'''

################################################# if issue with column geometry : 

radius = 12000 / 2  # Buffer radius in meters
buffer_geom = centroid.buffer(radius)

def calculate_metrics_adj(gdf, buffer_geom, diameter):
    metrics = {}
    print(f"\nCalculating metrics for buffer diameter: {diameter}m")
    land_surface = calculate_land_surface(gdf, buffer_geom)
    vegetation = calculate_vegetation(gdf, buffer_geom)
    urban_geometry = calculate_urban_geometry(gdf, buffer_geom)

    # Combine all metrics into a single dictionary
    metrics[diameter] = {**land_surface, **vegetation, **urban_geometry}
    print(f"Metrics for {diameter}m buffer: {metrics[diameter]}")

    return metrics

all_metrics = calculate_metrics_adj(gdf_combined, buffer_geom, 12000)

# Convert metrics dictionary to DataFrame
metrics_df = pd.DataFrame(all_metrics).T  # Transpose to have diameters as rows
metrics_df.index.name = 'Buffer_Diameter_m'

# Reset index to turn it into a column
metrics_df = metrics_df.reset_index()

# Display the metrics DataFrame
print("\nCalculated Metrics:")
print(metrics_df.head())

# Define the output path for the metrics CSV
output_csv_path = '/content/drive/MyDrive/wd/osmosis-0.49.2/osmo/output2/TAI_PO_results.csv' #### **

# Save the metrics to CSV
metrics_df.to_csv(output_csv_path, index=False)

print(f"\nMetrics saved to {output_csv_path}")

import matplotlib.pyplot as plt

def visualize_buffer(gdf, buffer_geom, diameter):
    fig, ax = plt.subplots(figsize=(10, 10))
    gdf.plot(ax=ax, color='blue', alpha=0.5, markersize=10, label='OSM Elements')
    gpd.GeoSeries([buffer_geom]).boundary.plot(ax=ax, color='red', label=f'{diameter}m Buffer')
    plt.title(f'Buffer {diameter}m around Centroid: TAI PO district') ### **
    plt.legend()
    plt.xlabel('Easting (meters)')
    plt.ylabel('Northing (meters)')
    output_folder = '/content/drive/MyDrive/wd/osmosis-0.49.2/osmo/output2'
    output_plot_path = os.path.join(output_folder, f'TAI_PO{diameter}m_buffer.png') #### **
    plt.savefig(output_plot_path)
    plt.close()
    print(f"Visualization saved to {output_plot_path}")

visualize_buffer(gdf_combined, buffer_geom, 12000)

'''
