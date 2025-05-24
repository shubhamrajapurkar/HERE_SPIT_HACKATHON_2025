import geopandas as gpd
import matplotlib.pyplot as plt

# --- Load the datasets ---
try:
    poi_file_path = "Singapore_Prime_LAT.shp" # Replace with actual path
    streets_file_path = "Streets.shp"         # Replace with actual path

    pois_gdf = gpd.read_file(poi_file_path)
    streets_gdf = gpd.read_file(streets_file_path)

    print("Successfully loaded POI and Street data.")

except Exception as e:
    print(f"Error loading shapefiles: {e}")
    # Exit or handle error appropriately
    exit()

# --- Inspect POI Data ---
print("\n--- POI Data Inspection ---")
print(f"POI CRS: {pois_gdf.crs}")
print("POI Head:\n", pois_gdf.head())
# print("POI Info:")
# pois_gdf.info()
print(f"POI Geometry Types: {pois_gdf.geom_type.unique()}")
print(f"Number of POIs: {len(pois_gdf)}")

# --- Inspect Street Data ---
print("\n--- Street Data Inspection ---")
print(f"Street CRS: {streets_gdf.crs}")
print("Street Head:\n", streets_gdf.head())
# print("Street Info:")
# streets_gdf.info()
print(f"Street Geometry Types: {streets_gdf.geom_type.unique()}")
print(f"Number of Street Segments: {len(streets_gdf)}")

# --- CRS Check and Reprojection (IMPORTANT) ---
if pois_gdf.crs != streets_gdf.crs:
    print(f"\nWarning: CRS mismatch! POIs: {pois_gdf.crs}, Streets: {streets_gdf.crs}")
    # Assuming you want to reproject streets to match POIs
    # This is a common step. Choose a suitable target CRS if both are different from what you need.
    print(f"Attempting to reproject streets to {pois_gdf.crs}...")
    try:
        streets_gdf = streets_gdf.to_crs(pois_gdf.crs)
        print(f"Streets reprojected. New CRS: {streets_gdf.crs}")
    except Exception as e:
        print(f"Error during reprojection: {e}")
        # Decide how to handle this: exit, or proceed with caution
else:
    print("\nCRS match. Good to proceed.")


# --- Basic Visualization ---
# Plot a sample to avoid overly dense plots if data is large
sample_pois = pois_gdf.sample(n=min(1000, len(pois_gdf)), random_state=42) # Plot up to 1000 POIs
# For streets, if it's very dense, you might want to plot a bounding box or a sample
# For now, let's try plotting all streets if it's not too overwhelming

fig, ax = plt.subplots(1, 1, figsize=(12, 10))
streets_gdf.plot(ax=ax, linewidth=0.5, color='gray', label='Streets')
sample_pois.plot(ax=ax, marker='o', color='red', markersize=5, label='Sample POIs')

# Get bounds from the street data or combined data for better zoom
# If streets_gdf is large, this might take time.
# Alternatively, define a specific area of interest if known.
minx, miny, maxx, maxy = streets_gdf.total_bounds
# Add a small buffer
buffer_x = (maxx - minx) * 0.05
buffer_y = (maxy - miny) * 0.05
ax.set_xlim(minx - buffer_x, maxx + buffer_x)
ax.set_ylim(miny - buffer_y, maxy + buffer_y)

plt.title("Initial Plot of Streets and Sample POIs")
plt.xlabel("Longitude / Easting")
plt.ylabel("Latitude / Northing")
plt.legend()
plt.tight_layout()
plt.show()

print("\n--- POI Column Names ---")
print(list(pois_gdf.columns))

print("\n--- Unique values in 'NEAR_Y' (sample) ---") # If too many, this might be slow
# Check a few common values or descriptive stats
print(pois_gdf['NEAR_Y'].value_counts(dropna=False).head(10))
print(pois_gdf['NEAR_Y'].describe())

# Look for columns that might contain POI Name, Address, or Category
# Example: (Replace 'NAME_COLUMN', 'ADDRESS_COLUMN', 'CATEGORY_COLUMN' with actual names if you find them)
# potential_name_cols = [col for col in pois_gdf.columns if 'NAME' in col.upper()]
# potential_addr_cols = [col for col in pois_gdf.columns if 'ADDR' in col.upper() or 'STREET' in col.upper()]
# potential_cat_cols = [col for col in pois_gdf.columns if 'CAT' in col.upper() or 'TYPE' in col.upper()]
# print(f"Potential Name Columns: {potential_name_cols}")
# print(f"Potential Address Columns: {potential_addr_cols}")
# print(f"Potential Category Columns: {potential_cat_cols}")

print("\n--- Street Column Names (First 20 and Last 20) ---") # 100 is too many to print all at once
street_cols = list(streets_gdf.columns)
print(street_cols[:20])
print("...")
print(street_cols[-20:])

# Try to find common street attribute names (you might need to guess based on vendor conventions)
# For example, HERE data often uses 'ST_NAME', 'FUNC_CLASS', 'DIR_TRAVEL'
# potential_street_name_cols = [col for col in streets_gdf.columns if 'NAME' in col.upper() or 'STN' in col.upper()]
# potential_road_class_cols = [col for col in streets_gdf.columns if 'CLASS' in col.upper() or 'FRC' in col.upper() or 'FOW' in col.upper() or 'TYPE' in col.upper()]
# potential_direction_cols = [col for col in streets_gdf.columns if 'DIR' in col.upper()]

# print(f"Potential Street Name Columns: {potential_street_name_cols}")
# print(f"Potential Road Class Columns: {potential_road_class_cols}")
# print(f"Potential Direction Columns: {potential_direction_cols}")

# If you identify a road class column, let's see its unique values:
# identified_road_class_col = 'YOUR_ROAD_CLASS_COLUMN_NAME' # Replace this
# if identified_road_class_col in streets_gdf.columns:
#     print(f"\n--- Unique values in '{identified_road_class_col}' ---")
#     print(streets_gdf[identified_road_class_col].value_counts(dropna=False))