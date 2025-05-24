import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# --- Parameters to Configure ---
FAULTY_POIS_FILE = "faulty_pois.gpkg"  # Your input faulty POIs GeoPackage file
# If your GeoPackage has multiple layers, you might need to specify the layer name when reading:
# e.g., faulty_pois_gdf = gpd.read_file(FAULTY_POIS_FILE, layer='your_poi_layer_name')
FAULTY_POIS_LAYER_NAME =  "flagged_pois" 
# Set to your layer name string if needed, e.g., 'faulty_points_layer'

BUILDINGS_FILE = "overpass_buildings.geojson"  # Your input OSM buildings GeoJSON
OUTPUT_RELOCATED_POIS_FILE = "relocated_pois_on_centroids.geojson"

# Field names (ADJUST THESE BASED ON YOUR ACTUAL DATA)
# Updated based on your provided image:
POI_ID_FIELD = "OBJECTID"          # Unique ID field in your POI data
POI_BUILDING_NAME_FIELD = "BUILDING_N" # Building name field in your POI data
OSM_BUILDING_NAME_FIELD = "name"       # Name field in your OSM building data

SEARCH_RADIUS_METERS = 75  # Search radius in meters to look for candidate buildings

# Coordinate Reference Systems
SOURCE_CRS = "EPSG:4326"  # Assuming input data might be in WGS84 (lat/lon)
# GeoPackage usually stores CRS info, so geopandas should pick it up.
# If not, this SOURCE_CRS will be used as an assumption if CRS is None.
TARGET_CRS = "EPSG:3414"  # Projected CRS for Singapore (e.g., SVY21) for accurate distances/centroids
OUTPUT_CRS = "EPSG:4326"  # Output GeoJSON will be in WGS84

# --- End of Configuration ---

def relocalize_pois():
    print(f"Loading faulty POIs from: {FAULTY_POIS_FILE}")
    try:
        if FAULTY_POIS_LAYER_NAME:
            print(f"  (Using layer: {FAULTY_POIS_LAYER_NAME})")
            faulty_pois_gdf = gpd.read_file(FAULTY_POIS_FILE, layer=FAULTY_POIS_LAYER_NAME)
        else:
            faulty_pois_gdf = gpd.read_file(FAULTY_POIS_FILE)
        
        if faulty_pois_gdf.crs is None:
            print(f"Warning: Faulty POIs GeoDataFrame has no CRS defined. Assuming {SOURCE_CRS}.")
            faulty_pois_gdf = faulty_pois_gdf.set_crs(SOURCE_CRS, allow_override=True)
        else:
            print(f"  Faulty POIs CRS detected: {faulty_pois_gdf.crs}")

    except Exception as e:
        print(f"Error loading faulty POIs from GeoPackage: {e}")
        print("Make sure the file path is correct and the GeoPackage is valid.")
        print("If the GeoPackage has multiple layers, set the FAULTY_POIS_LAYER_NAME variable.")
        return

    print(f"Loading building footprints from: {BUILDINGS_FILE}")
    try:
        buildings_gdf = gpd.read_file(BUILDINGS_FILE)
        if buildings_gdf.crs is None:
            print(f"Warning: Buildings GeoDataFrame has no CRS defined. Assuming {SOURCE_CRS}.")
            buildings_gdf = buildings_gdf.set_crs(SOURCE_CRS, allow_override=True)
        else:
            print(f"  Buildings CRS detected: {buildings_gdf.crs}")
    except Exception as e:
        print(f"Error loading building footprints: {e}")
        return

    # Ensure required fields exist
    if POI_ID_FIELD not in faulty_pois_gdf.columns:
        print(f"Error: POI ID field '{POI_ID_FIELD}' not found in faulty POIs data.")
        print(f"Available columns in faulty POIs: {faulty_pois_gdf.columns.tolist()}")
        return
    if POI_BUILDING_NAME_FIELD not in faulty_pois_gdf.columns:
        print(f"Warning: POI building name field '{POI_BUILDING_NAME_FIELD}' not found. Name matching will be skipped for all POIs.")
        print(f"Available columns in faulty POIs: {faulty_pois_gdf.columns.tolist()}")
        faulty_pois_gdf[POI_BUILDING_NAME_FIELD] = None # Add a dummy column if it doesn't exist
    if OSM_BUILDING_NAME_FIELD not in buildings_gdf.columns:
        print(f"Warning: OSM building name field '{OSM_BUILDING_NAME_FIELD}' not found in buildings data. Name matching will be less effective.")
        print(f"Available columns in buildings data: {buildings_gdf.columns.tolist()}")
        buildings_gdf[OSM_BUILDING_NAME_FIELD] = None # Add a dummy column if it doesn't exist


    # Convert to target CRS for metric operations
    print(f"Converting POIs to CRS: {TARGET_CRS}")
    faulty_pois_gdf = faulty_pois_gdf.to_crs(TARGET_CRS)
    print(f"Converting buildings to CRS: {TARGET_CRS}")
    buildings_gdf = buildings_gdf.to_crs(TARGET_CRS)

    # Ensure geometries are valid and we only work with Polygons/MultiPolygons for buildings
    buildings_gdf = buildings_gdf[buildings_gdf.geometry.is_valid]
    buildings_gdf = buildings_gdf[buildings_gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]

    if buildings_gdf.empty:
        print("No valid building polygons found after filtering. Exiting.")
        return

    # Create a spatial index for buildings for faster searching
    print("Creating spatial index for buildings...")
    buildings_sindex = buildings_gdf.sindex

    relocated_features_list = []
    processed_pois_count = 0
    relocated_count = 0

    print(f"Processing {len(faulty_pois_gdf)} faulty POIs...")
    for index, poi in faulty_pois_gdf.iterrows():
        processed_pois_count += 1
        if processed_pois_count % 100 == 0:
            print(f"  Processed {processed_pois_count}/{len(faulty_pois_gdf)} POIs...")

        original_poi_geom = poi.geometry
        poi_attributes = poi.to_dict()
        original_poi_id = poi_attributes.get(POI_ID_FIELD)
        
        # Extract POI building name safely, convert to lower string for matching
        # Handle potential NULL values which might be read as None or specific strings like 'NULL'
        poi_building_name_val = poi_attributes.get(POI_BUILDING_NAME_FIELD)
        if pd.notna(poi_building_name_val) and str(poi_building_name_val).strip().upper() != 'NULL':
            poi_building_name = str(poi_building_name_val).strip().lower()
        else:
            poi_building_name = None


        search_buffer = original_poi_geom.buffer(SEARCH_RADIUS_METERS)
        possible_matches_indices = list(buildings_sindex.intersection(search_buffer.bounds))
        candidate_buildings = buildings_gdf.iloc[possible_matches_indices]
        # Refine candidates by actual intersection with the buffer
        candidate_buildings = candidate_buildings[candidate_buildings.intersects(search_buffer)]

        matched_building_geom = None
        relocalization_method = None

        # --- Priority 1: Strict String Matching of Building Name ---
        if poi_building_name and not candidate_buildings.empty:
            named_match_candidates_list = [] # Use a list to store candidate rows
            for b_idx, building_candidate_row in candidate_buildings.iterrows():
                osm_building_name_val = building_candidate_row.get(OSM_BUILDING_NAME_FIELD)
                if pd.notna(osm_building_name_val) and str(osm_building_name_val).strip().upper() != 'NULL':
                    osm_b_name_lower = str(osm_building_name_val).strip().lower()
                    if osm_b_name_lower == poi_building_name:
                        named_match_candidates_list.append(building_candidate_row)
            
            if named_match_candidates_list:
                # Convert list of Series back to GeoDataFrame to easily get geometry and calculate distance
                named_match_candidates_gdf = gpd.GeoDataFrame(named_match_candidates_list, crs=candidate_buildings.crs)
                
                best_named_match_building = None
                min_dist_to_named_match = float('inf')

                for b_idx, named_match_building_row in named_match_candidates_gdf.iterrows():
                    dist = original_poi_geom.distance(named_match_building_row.geometry)
                    if dist < min_dist_to_named_match:
                        min_dist_to_named_match = dist
                        best_named_match_building = named_match_building_row
                
                if best_named_match_building is not None:
                    matched_building_geom = best_named_match_building.geometry
                    relocalization_method = "name_match"

        # --- Priority 2: Closest Building (Geometric) if no name match ---
        if matched_building_geom is None and not candidate_buildings.empty:
            closest_building_geom_overall = None
            min_dist_overall = float('inf')
            for b_idx, building_candidate_row in candidate_buildings.iterrows():
                dist = original_poi_geom.distance(building_candidate_row.geometry)
                if dist < min_dist_overall:
                    min_dist_overall = dist
                    closest_building_geom_overall = building_candidate_row.geometry
            
            if closest_building_geom_overall is not None:
                # No need to check min_dist_overall <= SEARCH_RADIUS_METERS here,
                # as candidate_buildings are already filtered by the search buffer.
                matched_building_geom = closest_building_geom_overall
                relocalization_method = "closest_geometric"

        # --- Perform Relocalization to Centroid ---
        if matched_building_geom:
            new_poi_location = matched_building_geom.centroid
            new_feature = {}
            # Copy original attributes, but ensure 'geometry' is not from original series
            for key, value in poi_attributes.items():
                if key != 'geometry': # Exclude the original geometry from attributes
                    new_feature[key] = value
            
            new_feature['geometry'] = new_poi_location
            new_feature['original_poi_id'] = original_poi_id # Keep track of original
            new_feature['reloc_method'] = relocalization_method
            
            relocated_features_list.append(new_feature)
            relocated_count += 1

    print(f"Finished processing. Total POIs: {len(faulty_pois_gdf)}, Relocated POIs: {relocated_count}")

    if relocated_features_list:
        relocated_gdf = gpd.GeoDataFrame(relocated_features_list, crs=TARGET_CRS)
        print(f"Converting relocated POIs to CRS: {OUTPUT_CRS}")
        relocated_gdf = relocated_gdf.to_crs(OUTPUT_CRS)
        
        print(f"Saving relocated POIs to: {OUTPUT_RELOCATED_POIS_FILE}")
        relocated_gdf.to_file(OUTPUT_RELOCATED_POIS_FILE, driver="GeoJSON")
        print("Done.")
    else:
        print("No POIs were relocated. No output file created.")

if __name__ == "__main__":
    relocalize_pois()