
import osmnx as ox
import folium

place = "Hong Kong, Hong Kong"
graph = ox.graph_from_place(place, network_type='drive')

# Convert graph to GeoDataFrames
nodes, edges = ox.graph_to_gdfs(graph)

# Save to GeoJSON
edges.to_file("/content/drive/MyDrive/wd/osm5_data/hong-kong-roads.geojson", driver='GeoJSON')

# Visualize roads
m_roads = folium.Map(location=[22.3193, 114.1694], zoom_start=11)
folium.GeoJson(edges).add_to(m_roads)
m_roads

# Retrieve buildings
tags = {'building': True}
buildings = ox.geometries_from_place(place, tags=tags)

#  Save buildings to GeoJSON
buildings.to_file("/content/drive/MyDrive/wd/osm5_data/hong-kong-buildings.geojson", driver='GeoJSON')

#  Visualize buildings
m_buildings = folium.Map(location=[22.3193, 114.1694], zoom_start=11)
folium.GeoJson(buildings).add_to(m_buildings)
m_buildings
