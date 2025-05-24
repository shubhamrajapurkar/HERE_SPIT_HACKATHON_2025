import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import folium
from folium.plugins import MarkerCluster
from shapely.geometry import Point, LineString, Polygon
import networkx as nx
from shapely.ops import polygonize, linemerge, unary_union

# Step 1: Load the street network shapefile
street_network = gpd.read_file('Streets.shp')

# Step 2: Check the CRS (Coordinate Reference System)
print("Original CRS:", street_network.crs)

# Step 3: Convert to a projected CRS if needed (for accurate distance measurements)
# Choose an appropriate UTM zone based on your area
projected_crs = "EPSG:3857"  # Web Mercator - adjust as needed
street_network = street_network.to_crs(projected_crs)

# Step 4: Explore basic information
print("\nDataset Basic Information:")
print(f"Number of street segments: {len(street_network)}")
print("\nGeometry types:")
print(street_network.geometry.type.value_counts())
print("\nAttribute columns:")
print(street_network.columns)

# Step 5: Examine attribute information (helpful for understanding road types)
print("\nSample of attribute data:")
print(street_network.head())

# Step 6: Extract potential roundabout candidates using geometric properties
# There are several approaches:

# 6.1: Graph-based approach using NetworkX
# Convert street segments to a graph
G = nx.Graph()
for idx, row in street_network.iterrows():
    if row.geometry.geom_type == 'LineString':
        # Add each segment with its endpoints as nodes
        start_point = (row.geometry.coords[0][0], row.geometry.coords[0][1])
        end_point = (row.geometry.coords[-1][0], row.geometry.coords[-1][1])
        G.add_edge(start_point, end_point, geometry=row.geometry)

# Find circular patterns in the graph
# (This is a simplified approach - you'll need to refine this)
cycles = nx.cycle_basis(G)
potential_roundabouts_cycles = [cycle for cycle in cycles if len(cycle) >= 3 and len(cycle) <= 10]

print(f"\nPotential roundabout candidates (cycle-based): {len(potential_roundabouts_cycles)}")

# 6.2: Geometry-based approach
# Detect closed loops from line segments
lines = list(street_network.geometry)
merged_lines = linemerge(lines)
boundaries = unary_union(merged_lines)
potential_polygons = list(polygonize(boundaries))

# Filter polygons by size and shape characteristics
min_area = 50  # minimum area in square meters
max_area = 10000  # maximum area in square meters
min_circularity = 0.6  # 1.0 is a perfect circle

roundabout_candidates = []
for polygon in potential_polygons:
    area = polygon.area
    perimeter = polygon.length
    
    # Calculate circularity (how close to a circle)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
    
    if (min_area < area < max_area) and circularity > min_circularity:
        roundabout_candidates.append(polygon)

print(f"\nPotential roundabout candidates (geometry-based): {len(roundabout_candidates)}")

# Step 7: Create GeoDataFrame from candidates
roundabout_gdf = gpd.GeoDataFrame({'geometry': roundabout_candidates}, crs=projected_crs)
roundabout_gdf['area'] = roundabout_gdf.geometry.area
roundabout_gdf['perimeter'] = roundabout_gdf.geometry.length
roundabout_gdf['circularity'] = 4 * np.pi * roundabout_gdf['area'] / (roundabout_gdf['perimeter'] ** 2)

# Step 8: Visualize with Matplotlib
fig, ax = plt.subplots(figsize=(12, 10))
street_network.plot(ax=ax, color='gray', linewidth=0.5)

if not roundabout_gdf.empty:
    roundabout_gdf.plot(ax=ax, color='red', alpha=0.5)

plt.title('Street Network with Potential Roundabouts')
plt.tight_layout()
plt.savefig('potential_roundabouts.png')
plt.show()

# Step 9: Visualize with Folium (interactive)
# Convert back to WGS84 for web mapping
street_network_wgs84 = street_network.to_crs("EPSG:4326")
roundabout_gdf_wgs84 = roundabout_gdf.to_crs("EPSG:4326") if not roundabout_gdf.empty else None

# Create a Folium map
map_center = [street_network_wgs84.geometry.centroid.y.mean(), 
              street_network_wgs84.geometry.centroid.x.mean()]
m = folium.Map(location=map_center, zoom_start=13, tiles='OpenStreetMap')

# Add street network
for _, row in street_network_wgs84.iterrows():
    if row.geometry.geom_type == 'LineString':
        folium.PolyLine(
            locations=[(lat, lon) for lon, lat in row.geometry.coords],
            color='blue',
            weight=2,
            opacity=0.7
        ).add_to(m)

# Add roundabout candidates
if roundabout_gdf_wgs84 is not None and not roundabout_gdf_wgs84.empty:
    for idx, row in roundabout_gdf_wgs84.iterrows():
        # Get polygon boundary as a list of coordinates
        if row.geometry.geom_type in ['Polygon', 'MultiPolygon']:
            # For multipolygons, we just take the first polygon
            if row.geometry.geom_type == 'MultiPolygon':
                polygon = row.geometry.geoms[0]
            else:
                polygon = row.geometry
                
            # Extract exterior coordinates
            coords = [(lat, lon) for lon, lat in polygon.exterior.coords]
            
            # Add polygon to map
            folium.Polygon(
                locations=coords,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.4,
                popup=f"Area: {row['area']:.1f} m², Circularity: {row['circularity']:.3f}"
            ).add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Save interactive map
m.save('roundabout_detection_interactive.html')
print("\nInteractive map saved as: roundabout_detection_interactive.html")

# Step 10: Add additional Folium features for better analysis
# Create a heatmap of roundabout candidate centroids
if roundabout_gdf_wgs84 is not None and not roundabout_gdf_wgs84.empty:
    # Create a new map for the heatmap
    m_heat = folium.Map(location=map_center, zoom_start=13)
    
    # Get centroids of roundabouts
    roundabout_centroids = roundabout_gdf_wgs84.copy()
    roundabout_centroids['geometry'] = roundabout_centroids.geometry.centroid
    
    # Add centroids as a heat map
    from folium.plugins import HeatMap
    heat_data = [[point.y, point.x] for point in roundabout_centroids.geometry]
    HeatMap(heat_data).add_to(m_heat)
    
    m_heat.save('roundabout_heatmap.html')
    print("Heatmap saved as: roundabout_heatmap.html")

# Step 11: Save candidates to GeoJSONL
# Convert back to WGS84 for standard GeoJSON if not already
if roundabout_gdf_wgs84 is None and not roundabout_gdf.empty:
    roundabout_gdf_wgs84 = roundabout_gdf.to_crs("EPSG:4326")

# Save to GeoJSONL (one feature per line)
if roundabout_gdf_wgs84 is not None and not roundabout_gdf_wgs84.empty:
    with open('roundabout_candidates.geojsonl', 'w') as f:
        for idx, row in roundabout_gdf_wgs84.iterrows():
            geojson_feature = {
                'type': 'Feature',
                'geometry': row.geometry.__geo_interface__,
                'properties': {
                    'area': float(row['area']),
                    'perimeter': float(row['perimeter']),
                    'circularity': float(row['circularity'])
                }
            }
            f.write(f"{pd.Series(geojson_feature).to_json()}\n")

    print("\nGeoJSONL file created: roundabout_candidates.geojsonl")
else:
    print("\nNo roundabout candidates found to save to GeoJSONL.")