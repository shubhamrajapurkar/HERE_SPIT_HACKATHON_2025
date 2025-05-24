import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd # For setting display options if needed

# --- Load the datasets (assuming they are not already in memory from previous steps) ---
# If they are already loaded and named pois_gdf, streets_gdf, you can skip this part.
try:
    poi_file_path = "Singapore_Prime_LAT.shp" # Replace with actual path
    streets_file_path = "Streets.shp"         # Replace with actual path

    pois_gdf = gpd.read_file(poi_file_path)
    streets_gdf = gpd.read_file(streets_file_path)
    print("Successfully loaded POI and Street data (or using existing ones).\n")

except Exception as e:
    print(f"Error loading shapefiles: {e}")
    exit()

# --- Action Items from previous analysis ---

print("--- POI Attribute Exploration ---")

# 1. REC_TYPE in POIs
print("\n1. Value counts for 'REC_TYPE' in POIs:")
if 'REC_TYPE' in pois_gdf.columns:
    print(pois_gdf['REC_TYPE'].value_counts(dropna=False))
else:
    print("'REC_TYPE' column not found in POIs.")

# 2. GEO_LEVEL in POIs
print("\n2. Value counts for 'GEO_LEVEL' in POIs:")
if 'GEO_LEVEL' in pois_gdf.columns:
    print(pois_gdf['GEO_LEVEL'].value_counts(dropna=False))
else:
    print("'GEO_LEVEL' column not found in POIs.")

# 3. Compare DISPLAY_LO/LA with geometry.x/y in POIs
print("\n3. Comparison of DISPLAY_LO/LA with geometry coordinates (sample of 5):")
if all(col in pois_gdf.columns for col in ['DISPLAY_LO', 'DISPLAY_LA']):
    # Create temporary columns for geometry x and y for easier comparison
    temp_pois_df = pois_gdf[['DISPLAY_LO', 'DISPLAY_LA']].copy()
    temp_pois_df['geometry_x'] = pois_gdf.geometry.x
    temp_pois_df['geometry_y'] = pois_gdf.geometry.y
    temp_pois_df['diff_lon'] = abs(temp_pois_df['DISPLAY_LO'] - temp_pois_df['geometry_x'])
    temp_pois_df['diff_lat'] = abs(temp_pois_df['DISPLAY_LA'] - temp_pois_df['geometry_y'])
    print(temp_pois_df[['DISPLAY_LO', 'geometry_x', 'diff_lon', 'DISPLAY_LA', 'geometry_y', 'diff_lat']].head())
    print(f"Max difference in Longitude: {temp_pois_df['diff_lon'].max()}")
    print(f"Max difference in Latitude: {temp_pois_df['diff_lat'].max()}")
    del temp_pois_df
else:
    print("'DISPLAY_LO' or 'DISPLAY_LA' columns not found in POIs.")


# 4. Describe NEAR_DIST, NEAR_X, NEAR_Y in POIs
print("\n4. Descriptive statistics for 'NEAR_DIST', 'NEAR_X', 'NEAR_Y' in POIs:")
near_cols = ['NEAR_DIST', 'NEAR_X', 'NEAR_Y']
if all(col in pois_gdf.columns for col in near_cols):
    print(pois_gdf[near_cols].describe())
    # Check how many have NEAR_Y == -1
    near_y_neg_one_count = (pois_gdf['NEAR_Y'] == -1).sum()
    print(f"\nNumber of POIs with NEAR_Y == -1: {near_y_neg_one_count}")
    if near_y_neg_one_count > 0:
        print("Corresponding NEAR_DIST and NEAR_X for these cases (sample):")
        print(pois_gdf[pois_gdf['NEAR_Y'] == -1][near_cols].head())
else:
    print("One or more 'NEAR_DIST', 'NEAR_X', 'NEAR_Y' columns not found in POIs.")


print("\n\n--- Street Attribute Exploration ---")

# 5. km_count in Streets
print("\n5. Descriptive statistics for 'km_count' in Streets:")
if 'km_count' in streets_gdf.columns:
    print(streets_gdf['km_count'].describe())
else:
    print("'km_count' column not found in Streets.")

# 6. FuncClass in Streets
print("\n6. Value counts for 'FuncClass' in Streets:")
if 'FuncClass' in streets_gdf.columns:
    # Pandas display options can be helpful for long lists of categories
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(streets_gdf['FuncClass'].value_counts(dropna=False))
else:
    print("'FuncClass' column not found in Streets.")

print("\n\n--- Reprojection ---")
# Target CRS for Singapore
target_crs_epsg = "EPSG:3414" # SVY21 / Singapore TM

# Reproject POIs
if pois_gdf.crs and pois_gdf.crs.to_string().upper() != target_crs_epsg:
    print(f"\nReprojecting POIs from {pois_gdf.crs} to {target_crs_epsg}...")
    pois_gdf_proj = pois_gdf.to_crs(target_crs_epsg)
    print(f"POIs reprojected. New CRS: {pois_gdf_proj.crs}")
    print(f"Original POI head (geometry):\n{pois_gdf.geometry.head()}")
    print(f"Projected POI head (geometry):\n{pois_gdf_proj.geometry.head()}")
else:
    pois_gdf_proj = pois_gdf.copy() # Already in target CRS or no CRS info
    print(f"POIs already in target CRS ({pois_gdf_proj.crs}) or no source CRS to compare.")

# Reproject Streets
if streets_gdf.crs and streets_gdf.crs.to_string().upper() != target_crs_epsg:
    print(f"\nReprojecting Streets from {streets_gdf.crs} to {target_crs_epsg}...")
    streets_gdf_proj = streets_gdf.to_crs(target_crs_epsg)
    print(f"Streets reprojected. New CRS: {streets_gdf_proj.crs}")
    print(f"Original Street head (geometry bounds):\n{streets_gdf.geometry.head().total_bounds}") # Printing full geometry is too much
    print(f"Projected Street head (geometry bounds):\n{streets_gdf_proj.geometry.head().total_bounds}")
else:
    streets_gdf_proj = streets_gdf.copy() # Already in target CRS or no CRS info
    print(f"Streets already in target CRS ({streets_gdf_proj.crs}) or no source CRS to compare.")

# --- Final Check: Verify data types and geometry after projection (optional but good) ---
print("\n--- Post-Reprojection Info ---")
print("Projected POIs Info:")
pois_gdf_proj.info(verbose=False, memory_usage='deep') # Concise info
print(f"Projected POI CRS: {pois_gdf_proj.crs}")
print(f"Number of projected POIs: {len(pois_gdf_proj)}")


print("\nProjected Streets Info:")
streets_gdf_proj.info(verbose=False, memory_usage='deep') # Concise info
print(f"Projected Street CRS: {streets_gdf_proj.crs}")
print(f"Number of projected Street Segments: {len(streets_gdf_proj)}")

# Target CRS for Singapore
target_crs = "EPSG:3414" # SVY21 / Singapore TM

if pois_gdf.crs.to_string() != target_crs:
    print(f"\nReprojecting POIs from {pois_gdf.crs} to {target_crs}...")
    pois_gdf_proj = pois_gdf.to_crs(target_crs)
    print(f"POIs reprojected. New CRS: {pois_gdf_proj.crs}")
else:
    pois_gdf_proj = pois_gdf.copy() # Already in target CRS
    print(f"POIs already in target CRS: {pois_gdf_proj.crs}")


if streets_gdf.crs.to_string() != target_crs:
    print(f"Reprojecting Streets from {streets_gdf.crs} to {target_crs}...")
    streets_gdf_proj = streets_gdf.to_crs(target_crs)
    print(f"Streets reprojected. New CRS: {streets_gdf_proj.crs}")
else:
    streets_gdf_proj = streets_gdf.copy() # Already in target CRS
    print(f"Streets already in target CRS: {streets_gdf_proj.crs}")


# From now on, use pois_gdf_proj and streets_gdf_proj


# --- Assuming pois_gdf_proj and streets_gdf_proj are already loaded and projected ---
# If not, uncomment and run the loading/projection part from your previous successful output.
# print(pois_gdf_proj.head()) # Quick check to ensure it's the projected data
# print(streets_gdf_proj.head()) # Quick check

print("\n--- Step 1: Recalculating Distance to Nearest Street & Getting Nearest Street Info ---")

# For sjoin_nearest, it's good practice if the right GeoDataFrame (streets) has a simple index.
# We also want to keep the original street attributes, so we'll join them.
# The 'index_right' column created by sjoin_nearest will refer to the index of streets_gdf_proj.

# Ensure there are no NaN geometries, which can cause issues
pois_gdf_proj = pois_gdf_proj[pois_gdf_proj.geometry.notna()]
streets_gdf_proj = streets_gdf_proj[streets_gdf_proj.geometry.notna()]

print(f"Number of POIs before sjoin_nearest: {len(pois_gdf_proj)}")
print(f"Number of Streets before sjoin_nearest: {len(streets_gdf_proj)}")

try:
    # Perform the spatial join.
    # `how='left'` keeps all POIs.
    # `distance_col='dist_to_street_m'` will name the new column containing distances.
    # sjoin_nearest needs GeoPandas >= 0.10.0
    pois_with_nearest_street = gpd.sjoin_nearest(
        pois_gdf_proj,
        streets_gdf_proj, # Note: No reset_index() here, sjoin_nearest handles it
        how='left',
        distance_col="dist_to_street_m" # This will create the distance column
    )

    print("\nSuccessfully performed sjoin_nearest.")
    print(f"Number of rows in joined table: {len(pois_with_nearest_street)}")
    print("Columns in joined table:", pois_with_nearest_street.columns)

    # Let's examine the new distance column
    if 'dist_to_street_m' in pois_with_nearest_street.columns:
        print("\nDescriptive statistics for 'dist_to_street_m':")
        print(pois_with_nearest_street['dist_to_street_m'].describe())

        # How many POIs have a large distance?
        threshold_distance = 50  # meters
        far_pois_count = (pois_with_nearest_street['dist_to_street_m'] > threshold_distance).sum()
        print(f"\nNumber of POIs > {threshold_distance}m from any street: {far_pois_count} (out of {len(pois_with_nearest_street)})")
        if len(pois_with_nearest_street) > 0: # Avoid division by zero
            print(f"Percentage of far POIs: {100 * far_pois_count / len(pois_with_nearest_street):.2f}%")

        # Check POIs that didn't get a distance (should be 0 if sjoin_nearest worked as expected with left join)
        # or have very large distances
        print("\nPOIs with potentially problematic distances (sample):")
        print(pois_with_nearest_street[
            (pois_with_nearest_street['dist_to_street_m'].isnull()) |
            (pois_with_nearest_street['dist_to_street_m'] > 1000) # e.g., POIs > 1km from a street
        ][['OBJECTID', 'STREET_NAM', 'dist_to_street_m', 'LINK_ID', 'ST_NAME']].head()) # Showing some POI and joined Street attributes

        # For future use, we often want the POI GeoDataFrame to have this new distance column
        # and perhaps key attributes of the NEAREST street.
        # The `pois_with_nearest_street` DataFrame currently has ONE ROW PER POI,
        # containing all original POI columns AND all columns from its NEAREST street.
        # This is often what we want. Let's rename it for clarity.
        pois_analyzed_gdf = pois_with_nearest_street

    else:
        print("\nError: 'dist_to_street_m' column was not created by sjoin_nearest.")
        print("This might happen if your GeoPandas version is older or if there was an unexpected issue.")
        # As a fallback, assign the original pois_gdf_proj to pois_analyzed_gdf and add a placeholder distance
        pois_analyzed_gdf = pois_gdf_proj.copy()
        pois_analyzed_gdf['dist_to_street_m'] = -1.0 # Placeholder


except AttributeError as e:
    print(f"\nAttributeError during sjoin_nearest: {e}")
    print("This often means `sjoin_nearest` is not available or `distance_col` is not a recognized parameter.")
    print("Your GeoPandas version might be older than 0.10.0. Please consider upgrading GeoPandas: `pip install geopandas --upgrade`")
    print("Skipping distance calculation for now. Assigning placeholder distance.")
    pois_analyzed_gdf = pois_gdf_proj.copy()
    pois_analyzed_gdf['dist_to_street_m'] = -1.0 # Placeholder
except Exception as e:
    print(f"\nAn unexpected error occurred during sjoin_nearest: {e}")
    pois_analyzed_gdf = pois_gdf_proj.copy()
    pois_analyzed_gdf['dist_to_street_m'] = -1.0 # Placeholder


print("\nSample of pois_analyzed_gdf (first 5 rows, selected columns):")
# Select some key POI columns and the new distance, and some key joined street columns
poi_cols_to_show = ['OBJECTID', 'STREET_NAM', 'GEO_LEVEL', 'geometry']
street_cols_to_show = ['LINK_ID', 'ST_NAME', 'PHYS_LANES'] # Example street cols
all_cols_to_show = [col for col in poi_cols_to_show if col in pois_analyzed_gdf.columns]
all_cols_to_show.append('dist_to_street_m') # Ensure distance is shown
all_cols_to_show.extend([col for col in street_cols_to_show if col in pois_analyzed_gdf.columns])

# Remove duplicates if any column name exists in both original POI and Street (e.g. 'geometry_left', 'geometry_right')
# For simplicity, let's just try to print. If there's an error, we'll adjust.
try:
    print(pois_analyzed_gdf[list(set(all_cols_to_show))].head()) # Use set to avoid duplicate column names if any
except KeyError as e:
    print(f"KeyError when trying to display sample: {e}. Available columns: {pois_analyzed_gdf.columns}")
    print(pois_analyzed_gdf.head())


print("\n--- Step 2: Exploring Alternative Street Type Indicators (from streets_gdf_proj) ---")

# We'll use the projected streets GeoDataFrame: streets_gdf_proj
# Because FuncClass was empty, we look for other clues.

street_attribute_candidates = ['ST_TYP_BEF', 'ST_TYP_AFT', 'EXPR_LANE', 'CARPOOLRD', 'PHYS_LANES']
# Add more column names from streets_gdf_proj.columns if they look promising for road type

print(f"All columns in streets_gdf_proj: {list(streets_gdf_proj.columns)}")

for col_name in street_attribute_candidates:
    if col_name in streets_gdf_proj.columns:
        print(f"\nValue counts for Street column '{col_name}':")
        # Check if the column is numeric or object/string for appropriate display
        if pd.api.types.is_numeric_dtype(streets_gdf_proj[col_name]):
            print(streets_gdf_proj[col_name].value_counts(dropna=False).sort_index().head(20)) # For numeric, sort by value
        else:
            print(streets_gdf_proj[col_name].value_counts(dropna=False).head(20)) # For object/string

        print(f"Number of NaNs in '{col_name}': {streets_gdf_proj[col_name].isnull().sum()} / {len(streets_gdf_proj)}")
    else:
        print(f"\nStreet column '{col_name}' not found in streets_gdf_proj.")

# Specific check for keywords in ST_NAME (Street Name)
if 'ST_NAME' in streets_gdf_proj.columns:
    print("\nChecking ST_NAME for keywords (e.g., EXPRESSWAY, HIGHWAY):")
    streets_gdf_proj['ST_NAME_upper'] = streets_gdf_proj['ST_NAME'].astype(str).str.upper() # Ensure string and uppercase

    expressway_streets_count = streets_gdf_proj[streets_gdf_proj['ST_NAME_upper'].str.contains('EXPRESSWAY', na=False)].shape[0]
    print(f"Number of street segments with 'EXPRESSWAY' in name: {expressway_streets_count}")

    highway_streets_count = streets_gdf_proj[streets_gdf_proj['ST_NAME_upper'].str.contains('HIGHWAY', na=False)].shape[0]
    print(f"Number of street segments with 'HIGHWAY' in name: {highway_streets_count}")

    avenue_streets_count = streets_gdf_proj[streets_gdf_proj['ST_NAME_upper'].str.contains('AVENUE', na=False)].shape[0]
    print(f"Number of street segments with 'AVENUE' in name: {avenue_streets_count}")

    road_streets_count = streets_gdf_proj[streets_gdf_proj['ST_NAME_upper'].str.contains('ROAD', na=False)].shape[0]
    print(f"Number of street segments with 'ROAD' in name: {road_streets_count}")

    lane_streets_count = streets_gdf_proj[streets_gdf_proj['ST_NAME_upper'].str.contains('LANE', na=False)].shape[0]
    print(f"Number of street segments with 'LANE' in name: {lane_streets_count}")

else:
    print("\n'ST_NAME' column not found in streets_gdf_proj.")


# Check for duplicates based on POI's original unique identifier
# Assuming 'OBJECTID' is the unique POI identifier
print(f"\nNumber of rows in pois_analyzed_gdf before deduplication: {len(pois_analyzed_gdf)}")
print(f"Number of unique POI OBJECTIDs: {pois_analyzed_gdf['OBJECTID'].nunique()}")

if len(pois_analyzed_gdf) > pois_analyzed_gdf['OBJECTID'].nunique():
    print("Duplicate POIs found due to sjoin_nearest. Deduplicating...")
    # Sort by distance (though for sjoin_nearest it should be the same for duplicates from the same POI)
    # then keep the first occurrence for each OBJECTID
    pois_analyzed_gdf = pois_analyzed_gdf.sort_values(by=['OBJECTID', 'dist_to_street_m'])
    pois_analyzed_gdf = pois_analyzed_gdf.drop_duplicates(subset=['OBJECTID'], keep='first')
    print(f"Number of rows after deduplication: {len(pois_analyzed_gdf)}")

print("\n--- Investigating Lane Information Further ---")
# Column names in pois_analyzed_gdf might have '_right' suffix if they came from streets_gdf_proj
# Let's find the actual column names for TO_LANES and FROM_LANES in pois_analyzed_gdf
to_lanes_col = None
from_lanes_col = None
phys_lanes_col = None # This would be the one from the street attributes

for col in pois_analyzed_gdf.columns:
    if 'TO_LANES' in col.upper(): # Catch 'TO_LANES' or 'TO_LANES_right' etc.
        to_lanes_col = col
    if 'FROM_LANES' in col.upper():
        from_lanes_col = col
    if 'PHYS_LANES' in col.upper() and col != 'PHYS_LANES': # Avoid original POI phys_lanes if it exists
        phys_lanes_col = col


if to_lanes_col:
    print(f"\nValue counts for '{to_lanes_col}' (from nearest street):")
    print(pois_analyzed_gdf[to_lanes_col].value_counts(dropna=False).head(10))
else:
    print("TO_LANES column not found in pois_analyzed_gdf.")

if from_lanes_col:
    print(f"\nValue counts for '{from_lanes_col}' (from nearest street):")
    print(pois_analyzed_gdf[from_lanes_col].value_counts(dropna=False).head(10))
else:
    print("FROM_LANES column not found in pois_analyzed_gdf.")

if phys_lanes_col: # This is the PHYS_LANES from the joined street data
    print(f"\nRe-checking '{phys_lanes_col}' (from nearest street) in pois_analyzed_gdf context:")
    print(pois_analyzed_gdf[phys_lanes_col].value_counts(dropna=False).head(10))
    # How many POIs are near streets with PHYS_LANES == 0?
    pois_near_zero_phys_lanes = pois_analyzed_gdf[pois_analyzed_gdf[phys_lanes_col] == 0]
    print(f"Number of POIs whose nearest street has {phys_lanes_col} == 0: {len(pois_near_zero_phys_lanes)}")
else:
    print("PHYS_LANES (from street) not clearly identified in pois_analyzed_gdf.")

print("\n--- Defining Flags for Potentially Misplaced POIs ---")
pois_analyzed_gdf['flag_far_from_street'] = False
pois_analyzed_gdf['flag_low_geo_level'] = False
pois_analyzed_gdf['flag_on_major_road_center'] = False # For the multi-digitized road scenario

# Rule 1: POIs far from any street
threshold_far_distance = 50  # meters
pois_analyzed_gdf.loc[pois_analyzed_gdf['dist_to_street_m'] > threshold_far_distance, 'flag_far_from_street'] = True
print(f"Number of POIs flagged 'far_from_street' (> {threshold_far_distance}m): {pois_analyzed_gdf['flag_far_from_street'].sum()}")

# Rule 2: POIs with low GEO_LEVEL
# GEO_LEVEL 0.0 was identified as potentially problematic. 7.0 is also lower precision.
low_geo_levels = [0.0, 7.0]
pois_analyzed_gdf.loc[pois_analyzed_gdf['GEO_LEVEL'].isin(low_geo_levels), 'flag_low_geo_level'] = True
print(f"Number of POIs flagged 'low_geo_level' (levels {low_geo_levels}): {pois_analyzed_gdf['flag_low_geo_level'].sum()}")

# Rule 3: POIs potentially on the centerline of a "major" road (multi-digitized road issue proxy)
# We need to define "major_road_lanes" based on findings from PHYS_LANES, TO_LANES, FROM_LANES.
# Let's tentatively assume PHYS_LANES (from street) is somewhat reliable for >0 values,
# or we can sum TO_LANES and FROM_LANES if they are per direction.

# Identify the correct lane column name from the joined street data
# It will likely end in '_right' if PHYS_LANES was also a POI column, or just be PHYS_LANES
# Based on your output "Columns in joined table", 'PHYS_LANES' itself is the one from streets.
street_phys_lanes_col = 'PHYS_LANES' # This is the one from the Streets table, now in pois_analyzed_gdf

if street_phys_lanes_col in pois_analyzed_gdf.columns:
    major_road_min_lanes = 4  # e.g., a road with 4 or more total lanes might be a candidate
    very_close_to_road_threshold = 1.0  # meters (e.g., within 1m of centerline)

    # Condition: POI is very close to a street that has many lanes
    pois_analyzed_gdf.loc[
        (pois_analyzed_gdf['dist_to_street_m'] < very_close_to_road_threshold) &
        (pois_analyzed_gdf[street_phys_lanes_col] >= major_road_min_lanes),
        'flag_on_major_road_center'
    ] = True
    print(f"Number of POIs flagged 'on_major_road_center' (dist < {very_close_to_road_threshold}m AND {street_phys_lanes_col} >= {major_road_min_lanes}): {pois_analyzed_gdf['flag_on_major_road_center'].sum()}")
else:
    print(f"Warning: Column '{street_phys_lanes_col}' for street lanes not found. Skipping 'flag_on_major_road_center'.")


# Combine flags: a POI is highly suspect if it triggers multiple flags
pois_analyzed_gdf['combined_flags_count'] = pois_analyzed_gdf[['flag_far_from_street', 'flag_low_geo_level', 'flag_on_major_road_center']].sum(axis=1)
print("\nCounts of POIs by number of flags triggered:")
print(pois_analyzed_gdf['combined_flags_count'].value_counts().sort_index())

print("\nSample of POIs with multiple flags (if any):")
print(pois_analyzed_gdf[pois_analyzed_gdf['combined_flags_count'] > 1][
    ['OBJECTID', 'STREET_NAM', 'GEO_LEVEL', 'dist_to_street_m', street_phys_lanes_col if street_phys_lanes_col in pois_analyzed_gdf.columns else 'OBJECTID', # ensure col exists
     'flag_far_from_street', 'flag_low_geo_level', 'flag_on_major_road_center', 'combined_flags_count']
].head(10))

print("\nOverall summary of flags:")
print(f"Total POIs: {len(pois_analyzed_gdf)}")
for flag_col in ['flag_far_from_street', 'flag_low_geo_level', 'flag_on_major_road_center']:
    if flag_col in pois_analyzed_gdf.columns:
        print(f"  {flag_col}: {pois_analyzed_gdf[flag_col].sum()}")

# --- Exporting flagged POIs for review ---
print("\n--- Exporting Flagged POIs for Review ---")

# Select POIs with at least one flag
# You might want to be more selective, e.g., combined_flags_count >= 2
pois_to_review = pois_analyzed_gdf[pois_analyzed_gdf['combined_flags_count'] >= 1].copy()

# Select relevant columns for review - adjust as needed
# Need to identify the actual name of the joined street's PHYS_LANES and ST_NAME
# Let's assume they are 'PHYS_LANES' and 'ST_NAME' from the sjoin output context
# (as the flagging code used 'PHYS_LANES' directly)

# List of desired POI columns
poi_core_cols = ['OBJECTID', 'CUSTOMER_I', 'FULL_POSTA', 'GEO_LEVEL', 'HOUSE_NUMB', 'BUILDING_N', 'STREET_NAM']
# List of desired joined street columns (these are now part of pois_analyzed_gdf)
street_joined_cols = ['LINK_ID', 'ST_NAME', 'PHYS_LANES', 'ST_TYP_AFT'] # Example, adjust if names have suffix
# Calculated/flag columns
analysis_cols = ['dist_to_street_m', 'flag_far_from_street', 'flag_low_geo_level', 'flag_on_major_road_center', 'combined_flags_count']

# Combine and ensure columns exist
export_columns = []
for col_list in [poi_core_cols, street_joined_cols, analysis_cols]:
    for col in col_list:
        if col in pois_to_review.columns:
            export_columns.append(col)
        # Check for common suffixes if direct match fails (e.g., _x, _y, _left, _right from joins)
        elif f"{col}_left" in pois_to_review.columns:
            export_columns.append(f"{col}_left")
        elif f"{col}_right" in pois_analyzed_gdf.columns:
             export_columns.append(f"{col}_right")


# Add geometry last
if 'geometry' in pois_to_review.columns:
    export_columns.append('geometry')

# Ensure unique columns
export_columns = list(dict.fromkeys(export_columns))


if not pois_to_review.empty:
    try:
        pois_to_review_export = pois_to_review[export_columns]

        # Export to GeoPackage (preferred for GIS as it keeps CRS and geometry)
        output_gpkg_path = "flagged_pois_for_review.gpkg"
        pois_to_review_export.to_file(output_gpkg_path, driver="GPKG", layer="flagged_pois")
        print(f"Exported {len(pois_to_review_export)} POIs for review to: {output_gpkg_path}")

        # Optionally, export to CSV (loses geometry unless you use WKT)
        # output_csv_path = "flagged_pois_for_review.csv"
        # pois_to_review_export_csv = pois_to_review_export.drop(columns=['geometry']) # Drop geometry for simple CSV
        # pois_to_review_export_csv.to_csv(output_csv_path, index=False)
        # print(f"Exported attributes of {len(pois_to_review_export_csv)} POIs for review to: {output_csv_path}")

    except Exception as e:
        print(f"Error during export: {e}")
        print("Please check column names in `export_columns` against `pois_to_review.columns`")
        print(f"Available columns: {list(pois_to_review.columns)}")

else:
    print("No POIs flagged for review based on current criteria.")

pois_analyzed_gdf['flag_street_name_mismatch'] = \
    (pois_analyzed_gdf['STREET_NAM'].astype(str).str.upper() != \
     pois_analyzed_gdf['ST_NAME'].astype(str).str.upper()) & \
    (pois_analyzed_gdf['STREET_NAM'].notna()) & \
    (pois_analyzed_gdf['ST_NAME'].notna())

# Then add this to combined_flags_count and your export
print(f"Number of POIs flagged 'street_name_mismatch': {pois_analyzed_gdf['flag_street_name_mismatch'].sum()}")