import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import polygonize, unary_union
import numpy as np
import matplotlib.pyplot as plt

def detect_roundabouts(roads_gdf, min_radius=10, max_radius=50, circularity_threshold=1, min_incoming=3):
    roads_gdf = roads_gdf[roads_gdf.is_valid].copy()
    roads_union = unary_union(roads_gdf.geometry)
    potential_roundabouts = list(polygonize(roads_union))
    candidates = []
    
    for poly in potential_roundabouts:
        center = poly.centroid
        radius = np.sqrt(poly.area / np.pi)
        circularity = (4 * np.pi * poly.area) / (poly.length ** 2)
        
        if not (min_radius <= radius <= max_radius and
                circularity >= circularity_threshold and
                poly.area >= np.pi * (min_radius**2)):
            continue
        
        # roads intersect the roundabout boundary
        edge_buffer = poly.buffer(2).boundary  # narrow buffer around edge
        intersecting_roads = roads_gdf[roads_gdf.geometry.intersects(edge_buffer)]
        incoming_roads = intersecting_roads[~intersecting_roads.geometry.within(poly.buffer(-1))]
        
        if len(incoming_roads) >= min_incoming:
            candidates.append({
                'geometry': poly,
                'radius': radius,
                'circularity': circularity,
                'center': center,
                'incoming_streets': len(incoming_roads)
            })
    
    return gpd.GeoDataFrame(candidates, crs=roads_gdf.crs)

def add_roundabout_attribute(roads_gdf, roundabouts_gdf, buffer_distance=5):
    """
    Add a roundabout attribute to the roads layer indicating if each street
    intersects with or is near a roundabout.
    
    Parameters:
    - roads_gdf: GeoDataFrame of roads/streets
    - roundabouts_gdf: GeoDataFrame of detected roundabouts
    - buffer_distance: Distance to buffer roundabouts for intersection check
    
    Returns:
    - roads_gdf with new 'has_roundabout' column
    """
    # Create a copy to avoid modifying original
    roads_with_attr = roads_gdf.copy()
    
    # Initialize the roundabout attribute column as boolean
    roads_with_attr['has_roundabout'] = False  # False = No roundabout, True = Has roundabout
    
    if len(roundabouts_gdf) == 0:
        print("No roundabouts detected - all streets marked as False")
        return roads_with_attr
    
    # Buffer roundabouts slightly to catch nearby streets
    buffered_roundabouts = roundabouts_gdf.geometry.buffer(buffer_distance)
    
    # Create a union of all buffered roundabouts for efficient intersection
    roundabouts_union = unary_union(buffered_roundabouts)
    
    # Check which roads intersect with any roundabout
    intersects_roundabout = roads_with_attr.geometry.intersects(roundabouts_union)
    
    # Update the attribute: True for streets that intersect roundabouts, False for others
    roads_with_attr.loc[intersects_roundabout, 'has_roundabout'] = True
    
    # Print summary statistics
    total_streets = len(roads_with_attr)
    streets_with_roundabouts = sum(roads_with_attr['has_roundabout'])
    
    print(f"Total streets: {total_streets}")
    print(f"Streets with roundabouts: {streets_with_roundabouts}")
    print(f"Streets without roundabouts: {total_streets - streets_with_roundabouts}")
    print(f"Percentage with roundabouts: {streets_with_roundabouts/total_streets*100:.1f}%")
    
    return roads_with_attr

# Load and project roads
roads = gpd.read_file("data/raw/Streets.shp").to_crs("EPSG:3059")

# Detect roundabouts
roundabouts = detect_roundabouts(roads, min_radius=10, max_radius=300, 
                                circularity_threshold=0.95, min_incoming=3)

# Add roundabout attribute to roads
roads_with_roundabouts = add_roundabout_attribute(roads, roundabouts, buffer_distance=10)

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: Original detection
roads.plot(ax=ax1, color='gray', linewidth=0.5, alpha=0.7)
roundabouts.plot(ax=ax1, color='red', alpha=0.6)
ax1.set_title(f"Detected Roundabouts: {len(roundabouts)}")

# Plot 2: Streets colored by roundabout attribute
streets_no_roundabout = roads_with_roundabouts[roads_with_roundabouts['has_roundabout'] == False]
streets_with_roundabout = roads_with_roundabouts[roads_with_roundabouts['has_roundabout'] == True]

streets_no_roundabout.plot(ax=ax2, color='gray', linewidth=0.5, alpha=0.7, label='No roundabout')
streets_with_roundabout.plot(ax=ax2, color='blue', linewidth=1, label='Has roundabout')
roundabouts.plot(ax=ax2, color='red', alpha=0.6, label='Roundabouts')

ax2.set_title("Streets Classified by Roundabout Presence")
ax2.legend()

plt.tight_layout()
plt.show()

# Save the updated roads layer with roundabout attribute
roads_with_roundabouts.to_file("roads_with_roundabout_attribute.gpkg", 
                              layer='roads_classified', driver="GPKG")

# Also save roundabouts separately
roundabouts.to_file("detected_roundabouts.gpkg", 
                   layer='roundabouts', driver="GPKG")

print("\nFiles saved:")
print("- roads_with_roundabout_attribute.gpkg (roads with has_roundabout column)")
print("- detected_roundabouts.gpkg (detected roundabouts)")

# Optional: Create a more detailed classification
def add_detailed_roundabout_attribute(roads_gdf, roundabouts_gdf):
    """
    Add a more detailed roundabout classification:
    0 = No roundabout nearby
    1 = Within roundabout area
    2 = Intersects with roundabout boundary
    3 = Near roundabout (within buffer)
    """
    roads_detailed = roads_gdf.copy()
    roads_detailed['roundabout_type'] = 0
    
    if len(roundabouts_gdf) == 0:
        return roads_detailed
    
    for idx, roundabout in roundabouts_gdf.iterrows():
        # Streets within the roundabout
        within_mask = roads_detailed.geometry.within(roundabout.geometry)
        roads_detailed.loc[within_mask, 'roundabout_type'] = 1
        
        # Streets intersecting the boundary
        intersects_mask = roads_detailed.geometry.intersects(roundabout.geometry.boundary)
        roads_detailed.loc[intersects_mask & (roads_detailed['roundabout_type'] == 0), 'roundabout_type'] = 2
        
        # Streets near the roundabout (buffered)
        near_mask = roads_detailed.geometry.intersects(roundabout.geometry.buffer(15))
        roads_detailed.loc[near_mask & (roads_detailed['roundabout_type'] == 0), 'roundabout_type'] = 3
    
    return roads_detailed

# Create detailed classification (optional)
roads_detailed = add_detailed_roundabout_attribute(roads, roundabouts)
roads_detailed.to_file("roads_detailed_roundabout_classification.gpkg", 
                      layer='roads_detailed', driver="GPKG")