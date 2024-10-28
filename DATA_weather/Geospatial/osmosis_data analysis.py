import osmium
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import os
import matplotlib.pyplot as plt
import ast

# Define directories
input_dir = '/content/drive/MyDrive/wd/osmosis-0.49.2/osmo/output_osm_pbf'
output_dir = '/content/drive/MyDrive/wd/osmosis-0.49.2/osmo/output'

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

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

def process_pbf_file(pbf_path):
    print(f"Processing file: {pbf_path}")
    district_name = os.path.basename(pbf_path).replace('.osm.pbf', '')

    # Initialize handler
    handler = OSMHandler()
    handler.apply_file(pbf_path, locations=True)

    # Extracted Data
    nodes_data = handler.nodes
    ways_data = handler.ways

    # Convert ways_data to DataFrame
    ways_df = pd.DataFrame(ways_data)

    # Create DataFrame for nodes
    nodes_df = pd.DataFrame.from_dict(nodes_data, orient='index', columns=['lon', 'lat'])
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

    # Project to Web Mercator (meters) (geographic to projected coordinates)
    gdf_combined = gdf_combined.to_crs(epsg=3857)

    # Define buffer diameters in meters
    diameters = [1000, 7000]

    # Calculate the centroid of all geometries in the district
    centroid = gdf_combined.geometry.unary_union.centroid

    # Create buffer geometries
    buffers = {}
    for diameter in diameters:
        radius = diameter / 2  # Buffer radius in meters
        buffer_geom = centroid.buffer(radius)
        buffers[diameter] = buffer_geom

    # Function definitions for calculating metrics (land surface, vegetation, urban geometry) omitted for brevity
    # Include your `calculate_land_surface`, `calculate_vegetation`, `calculate_urban_geometry` functions here...

    # Calculate metrics for all buffer zones
    all_metrics = calculate_metrics(gdf_combined, buffers)

    # Convert metrics dictionary to DataFrame
    metrics_df = pd.DataFrame(all_metrics).T  # Transpose to have diameters as rows
    metrics_df.index.name = 'Buffer_Diameter_m'
    metrics_df = metrics_df.reset_index()

    # Define output paths for metrics CSV and visualization images
    output_csv_path = os.path.join(output_dir, f'{district_name}_metrics.csv')

    # Save metrics to CSV
    metrics_df.to_csv(output_csv_path, index=False)
    print(f"Metrics saved to {output_csv_path}")

    # Visualization for each buffer
    for diameter, buffer_geom in buffers.items():
        fig, ax = plt.subplots(figsize=(10, 10))
        gdf_combined.plot(ax=ax, color='blue', alpha=0.5, markersize=10, label='OSM Elements')
        gpd.GeoSeries([buffer_geom]).boundary.plot(ax=ax, color='red', label=f'{diameter}m Buffer')
        plt.title(f'Buffer {diameter}m around Centroid - {district_name}')
        plt.legend()
        plt.xlabel('Easting (meters)')
        plt.ylabel('Northing (meters)')
        plt_path = os.path.join(output_dir, f'{district_name}_buffer_{diameter}m.png')
        plt.savefig(plt_path)
        plt.close()
        print(f"Visualization saved to {plt_path}")

# Iterate through all PBF files in the input directory
for pbf_file in os.listdir(input_dir):
    if pbf_file.endswith('.osm.pbf'):
        pbf_path = os.path.join(input_dir, pbf_file)
        process_pbf_file(pbf_path)
