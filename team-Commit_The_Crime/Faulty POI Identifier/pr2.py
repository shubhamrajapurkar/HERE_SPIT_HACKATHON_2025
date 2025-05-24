import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.ops import nearest_points
# from shapely.geometry import Point, LineString # Not directly used, but good for context
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import traceback  # For detailed error printing

# --- Configuration ---
GPKG_PATH_FOR_POINTS = "flagged_pois_for_review.gpkg"
POINTS_LAYER_NAME = "flagged_pois"
ROADS_SHAPEFILE_PATH = "Streets.shp"
BUILDINGS_SHAPEFILE_PATH = "Your_Buildings.shp"  # Placeholder

TARGET_PROJECTED_CRS = "EPSG:3414"
OUTPUT_LABELING_GPKG_PATH = "points_for_manual_labeling.gpkg"

# --- Load Data ---
print("--- Loading Initial Data ---")
try:
    points_gdf = gpd.read_file(GPKG_PATH_FOR_POINTS, layer=POINTS_LAYER_NAME)
    print(
        f"Loaded {len(points_gdf)} potentially misplaced points from '{GPKG_PATH_FOR_POINTS}'. Original CRS: {points_gdf.crs}")

    roads_gdf = gpd.read_file(ROADS_SHAPEFILE_PATH)
    print(f"Loaded {len(roads_gdf)} road segments from '{ROADS_SHAPEFILE_PATH}'. Original CRS: {roads_gdf.crs}")

    buildings_gdf = None
    if BUILDINGS_SHAPEFILE_PATH and BUILDINGS_SHAPEFILE_PATH.lower() != "your_buildings.shp":
        try:
            buildings_gdf = gpd.read_file(BUILDINGS_SHAPEFILE_PATH)
            if buildings_gdf is not None and not buildings_gdf.empty:
                print(
                    f"Loaded {len(buildings_gdf)} buildings from '{BUILDINGS_SHAPEFILE_PATH}'. Original CRS: {buildings_gdf.crs}")
            else:
                buildings_gdf = None
        except Exception as e:
            print(f"Warning: Could not load buildings shapefile '{BUILDINGS_SHAPEFILE_PATH}': {e}")
            buildings_gdf = None
    else:
        print(
            f"Building shapefile path ('{BUILDINGS_SHAPEFILE_PATH}') not provided or is placeholder; buildings data will be skipped.")

    # --- CRS Management ---
    print(f"\n--- Ensuring all data is in Target CRS: {TARGET_PROJECTED_CRS} ---")
    # Temporary list to hold GDFs and their names for processing
    gdfs_to_process = []
    if points_gdf is not None and not points_gdf.empty: gdfs_to_process.append((points_gdf, "Points"))
    if roads_gdf is not None and not roads_gdf.empty: gdfs_to_process.append((roads_gdf, "Roads"))
    if buildings_gdf is not None and not buildings_gdf.empty: gdfs_to_process.append((buildings_gdf, "Buildings"))

    processed_gdfs = {}  # To store processed GDFs

    for gdf_input, name in gdfs_to_process:
        gdf_temp = gdf_input.copy()  # Work on a copy
        if gdf_temp.crs is None:
            print(f"Warning: {name} layer has no CRS. Assuming geographic (EPSG:4326) and reprojecting.")
            gdf_temp.set_crs("EPSG:4326", inplace=True, allow_override=True)
            gdf_temp = gdf_temp.to_crs(TARGET_PROJECTED_CRS)
        elif gdf_temp.crs.to_string().upper() != TARGET_PROJECTED_CRS:
            print(f"Reprojecting {name} from {gdf_temp.crs} to {TARGET_PROJECTED_CRS}")
            gdf_temp = gdf_temp.to_crs(TARGET_PROJECTED_CRS)
        else:
            print(f"{name} are already in {TARGET_PROJECTED_CRS}")

        # Ensure active geometry column is named 'geometry'
        if gdf_temp.geometry.name != 'geometry':
            gdf_temp = gdf_temp.rename_geometry('geometry')

        processed_gdfs[name] = gdf_temp

    points_gdf = processed_gdfs.get("Points", points_gdf)  # Use processed or original if not in dict
    roads_gdf = processed_gdfs.get("Roads", roads_gdf)
    buildings_gdf = processed_gdfs.get("Buildings", buildings_gdf)


except Exception as e:
    print(f"Error during data loading or initial CRS transformation: {e}")
    traceback.print_exc()
    exit()

# --- Point ID and Indexing ---
if 'point_id' not in points_gdf.columns:
    if 'OBJECTID' in points_gdf.columns and points_gdf['OBJECTID'].is_unique:
        points_gdf['point_id'] = points_gdf['OBJECTID']
        print("\nUsed 'OBJECTID' as 'point_id'.")
    else:
        points_gdf = points_gdf.reset_index(drop=True)
        points_gdf['point_id'] = points_gdf.index
        print("\nCreated new sequential 'point_id'.")
else:
    print(f"\n'point_id' column already exists.")
    if not points_gdf['point_id'].is_unique:
        print("Warning: Existing 'point_id' is not unique. Resetting index and 'point_id'.")
        points_gdf = points_gdf.reset_index(drop=True)
        points_gdf['point_id'] = points_gdf.index

points_gdf = points_gdf.set_index('point_id', drop=False).rename_axis('point_id_idx')
print(f"Points GDF Index name: {points_gdf.index.name}, Is Unique: {points_gdf.index.is_unique}")

# --- Feature Engineering ---
print("\n--- Starting Optimized Feature Engineering ---")

# Pre-calculate buffered roads union
print("Pre-calculating buffered roads union...")
road_buffer_dist = 1.0
buffered_roads_union_geom = None
if not roads_gdf.empty and 'geometry' in roads_gdf.columns and not roads_gdf.geometry.is_empty.all():
    try:
        buffered_roads_union_geom = roads_gdf.buffer(road_buffer_dist).union_all()
        print("Buffered roads union calculated.")
    except Exception as e:
        print(f"Error calculating buffered roads union: {e}")
else:
    print("Roads GDF empty or geometry invalid; skipping buffer union.")

# Initialize feature columns
points_gdf['dist_to_nearest_road'] = np.nan
points_gdf['dist_to_projected_on_road'] = np.nan
points_gdf['dist_to_nearest_building'] = np.nan
points_gdf['is_on_road_buffer'] = 0
points_gdf['nearby_points_density'] = 0

# Vectorized nearest road calculation
if not points_gdf.empty and not roads_gdf.empty:
    print("Calculating nearest roads...")
    try:
        roads_for_join = roads_gdf.copy()
        roads_for_join['road_original_index'] = range(len(roads_for_join))

        # Ensure geometry names are 'geometry' for sjoin inputs
        # The GDFs should already have 'geometry' as active geometry name from CRS block

        points_with_nearest_road_info = gpd.sjoin_nearest(
            points_gdf,  # Active geom should be 'geometry', index 'point_id_idx'
            roads_for_join,  # Active geom should be 'geometry', has 'road_original_index'
            how='left',
            distance_col="temp_dist_to_road",
            lsuffix='pointgeom',
            rsuffix='roadgeom'
        )
        print("Nearest road sjoin completed.")

        # Deduplicate based on original point index (which is the index of points_with_nearest_road_info)
        points_with_nearest_road_info = points_with_nearest_road_info.sort_values(
            by=[points_with_nearest_road_info.index.name, 'temp_dist_to_road'])
        points_with_nearest_road_info = points_with_nearest_road_info[
            ~points_with_nearest_road_info.index.duplicated(keep='first')]

        print("Calculating projected distances to nearest roads...")
        processed_count = 0

        # The active geometry of the sjoin result *is* the geometry from the left (points_gdf).
        # If points_gdf's geometry was 'geometry', and roads_for_join's was 'geometry',
        # and suffixes were used, the left geometry in points_with_nearest_road_info
        # would be 'geometry_pointgeom'. If no suffix needed for left, it's 'geometry'.
        # The safest is to rely on points_with_nearest_road_info.geometry.name

        left_sjoin_geom_col_name = points_with_nearest_road_info.geometry.name

        for original_point_idx_val, row_sjoined in points_with_nearest_road_info.iterrows():
            point_geom_active_in_sjoin = row_sjoined[left_sjoin_geom_col_name]

            if pd.notna(row_sjoined.get('road_original_index')):
                try:
                    road_idx = int(row_sjoined['road_original_index'])
                    # Get road geometry from original roads_for_join using its active geometry column name
                    nearest_road_actual_geom = roads_for_join.loc[road_idx, roads_for_join.geometry.name]

                    if point_geom_active_in_sjoin and nearest_road_actual_geom and \
                            point_geom_active_in_sjoin.is_valid and nearest_road_actual_geom.is_valid:

                        projected_points_tuple = nearest_points(point_geom_active_in_sjoin, nearest_road_actual_geom)
                        projected_point_on_road = projected_points_tuple[1]
                        dist_projected = point_geom_active_in_sjoin.distance(projected_point_on_road)

                        points_gdf.loc[original_point_idx_val, 'dist_to_nearest_road'] = row_sjoined[
                            'temp_dist_to_road']
                        points_gdf.loc[original_point_idx_val, 'dist_to_projected_on_road'] = dist_projected
                    else:
                        points_gdf.loc[original_point_idx_val, 'dist_to_nearest_road'] = row_sjoined[
                            'temp_dist_to_road']
                        points_gdf.loc[original_point_idx_val, 'dist_to_projected_on_road'] = np.nan
                except KeyError:
                    points_gdf.loc[original_point_idx_val, 'dist_to_nearest_road'] = np.nan
                    points_gdf.loc[original_point_idx_val, 'dist_to_projected_on_road'] = np.nan
                except Exception as proj_e:
                    print(f"  Error processing point {original_point_idx_val} for projected distance: {proj_e}")
                    points_gdf.loc[original_point_idx_val, 'dist_to_nearest_road'] = np.nan
                    points_gdf.loc[original_point_idx_val, 'dist_to_projected_on_road'] = np.nan
            else:
                points_gdf.loc[original_point_idx_val, 'dist_to_nearest_road'] = np.nan
                points_gdf.loc[original_point_idx_val, 'dist_to_projected_on_road'] = np.nan

            processed_count += 1
            if processed_count % 500 == 0:
                print(
                    f"  Processed projected distance for {processed_count}/{len(points_with_nearest_road_info)} points.")
        print("Projected distances to nearest roads calculated.")
    except Exception as e:
        print(f"MAJOR Error during vectorized nearest road calculation: {e}")
        traceback.print_exc()
else:
    print("Skipped nearest road calculation (empty points or roads GDF).")

# Vectorized nearest building calculation
if buildings_gdf is not None and not buildings_gdf.empty and not points_gdf.empty:
    print("Calculating nearest buildings...")
    try:
        cols_to_drop_for_bldg_join = [
            col for col in points_gdf.columns if
            any(suffix in col for suffix in
                ['_road', '_roadgeom', '_pointgeom']) or  # Drop suffixed cols from previous join
            col in ['temp_dist_to_road', 'road_original_index', 'dist_to_nearest_building']  # Specific cols
        ]
        points_for_bldg_join = points_gdf.drop(columns=cols_to_drop_for_bldg_join, errors='ignore')
        # Ensure points_for_bldg_join still has its original geometry as active and named 'geometry'
        # points_gdf's geometry should already be 'geometry'
        if points_for_bldg_join.geometry.name != points_gdf.geometry.name:  # If drop changed active geom
            points_for_bldg_join = points_for_bldg_join.set_geometry(points_gdf.geometry.name)
        if points_for_bldg_join.geometry.name != 'geometry':
            points_for_bldg_join = points_for_bldg_join.rename_geometry('geometry')

        buildings_for_join = buildings_gdf.reset_index(drop=True)
        if buildings_for_join.geometry.name != 'geometry':  # buildings_gdf geom name was set earlier
            buildings_for_join = buildings_for_join.rename_geometry('geometry')

        points_with_nearest_building_info = gpd.sjoin_nearest(
            points_for_bldg_join,
            buildings_for_join,
            how='left',
            distance_col="temp_dist_to_building",
            lsuffix='pointgeom', rsuffix='bldggeom'  # Consistent suffixes
        )
        print("Nearest building sjoin completed.")

        if not points_with_nearest_building_info.empty:
            points_with_nearest_building_info = points_with_nearest_building_info.sort_values(
                by=[points_with_nearest_building_info.index.name, 'temp_dist_to_building'])
            points_with_nearest_building_info = points_with_nearest_building_info[
                ~points_with_nearest_building_info.index.duplicated(keep='first')]

            min_building_distances = points_with_nearest_building_info.groupby(
                points_with_nearest_building_info.index  # This index is from points_for_bldg_join (i.e., point_id_idx)
            )['temp_dist_to_building'].min()
            points_gdf['dist_to_nearest_building'] = points_gdf.index.map(min_building_distances)
        print("Distances to nearest buildings calculated.")
    except Exception as e:
        print(f"Error during vectorized nearest building calculation: {e}")
        traceback.print_exc()
else:
    print("Skipped nearest building calculation.")

# Loop for remaining features (buffer intersection, density)
print("Calculating buffer intersection and density...")
if not points_gdf.empty: points_gdf.sindex

loop_counter = 0
for original_point_idx_val, point_row in points_gdf.iterrows():
    point_geom = point_row.geometry  # This is the active geometry of points_gdf

    if buffered_roads_union_geom is not None:
        if point_geom and point_geom.is_valid and hasattr(point_geom, 'intersects'):
            if point_geom.intersects(buffered_roads_union_geom):
                points_gdf.loc[original_point_idx_val, 'is_on_road_buffer'] = 1
        elif point_geom is None or not point_geom.is_valid:
            pass

    if point_geom and point_geom.is_valid and hasattr(point_geom, 'buffer'):
        search_radius = 50
        current_point_buffer = point_geom.buffer(search_radius)
        try:  # Add try-except for sindex query if geometries are problematic
            possible_matches_index = list(points_gdf.sindex.intersection(current_point_buffer.bounds))
            if possible_matches_index:  # Only proceed if sindex returns something
                possible_matches = points_gdf.iloc[possible_matches_index]
                precise_matches = possible_matches[possible_matches.intersects(current_point_buffer)]
                points_gdf.loc[original_point_idx_val, 'nearby_points_density'] = len(precise_matches) - 1
            else:
                points_gdf.loc[original_point_idx_val, 'nearby_points_density'] = 0  # No potential matches from sindex
        except Exception as sindex_e:
            # print(f"  Warning: Error during sindex query for point {original_point_idx_val}: {sindex_e}")
            points_gdf.loc[original_point_idx_val, 'nearby_points_density'] = 0  # Default if sindex fails
    elif point_geom is None or not point_geom.is_valid:
        pass

    loop_counter += 1
    if loop_counter % 1000 == 0:
        print(f"  Processed buffer/density for {loop_counter}/{len(points_gdf)} points.")
print("Buffer intersection and density calculations complete.")

# --- Finalize features ---
points_with_features_gdf = points_gdf.copy()
feature_columns_to_fill = [
    'dist_to_nearest_road', 'dist_to_projected_on_road',
    'dist_to_nearest_building', 'is_on_road_buffer', 'nearby_points_density'
]
for col in feature_columns_to_fill:
    if col in points_with_features_gdf.columns and pd.api.types.is_numeric_dtype(points_with_features_gdf[col]):
        if points_with_features_gdf[col].isnull().all():
            points_with_features_gdf[col] = points_with_features_gdf[col].fillna(0)
            print(f"Warning: Column '{col}' was all NaN, filled with 0.")
        else:
            points_with_features_gdf[col] = points_with_features_gdf[col].fillna(points_with_features_gdf[col].mean())

print("\n--- Points GDF with all calculated features (first 3 rows) ---")
cols_to_show_final = ['point_id', 'geometry'] + [col for col in feature_columns_to_fill if
                                                 col in points_with_features_gdf.columns]
# Ensure columns to show actually exist to prevent KeyErrors
cols_to_show_final_existing = [c for c in cols_to_show_final if c in points_with_features_gdf.columns]
print(points_with_features_gdf.head(3)[cols_to_show_final_existing])

# --- HEURISTIC LABELING ---
print("\n--- Applying Heuristic Labeling ---")


def label_misplaced_heuristic(row):
    dist_check = row.get('dist_to_projected_on_road', float('inf'))
    if pd.isna(dist_check): dist_check = float('inf')
    if dist_check > 20:
        return 1
    return 0


points_with_features_gdf['is_misplaced'] = points_with_features_gdf.apply(label_misplaced_heuristic, axis=1)
print(f"Heuristically labeled {points_with_features_gdf['is_misplaced'].sum()} points.")
print("Value counts for 'is_misplaced' (heuristic):")
print(points_with_features_gdf['is_misplaced'].value_counts(dropna=False))

# --- EXPORT FOR MANUAL LABELING ---
print(f"\n--- Exporting data to '{OUTPUT_LABELING_GPKG_PATH}' for manual labeling review ---")
original_poi_attributes = ['OBJECTID', 'CUSTOMER_I', 'FULL_POSTA', 'GEO_LEVEL', 'HOUSE_NUMB', 'BUILDING_N',
                           'STREET_NAM']
export_cols = ['point_id']
for orig_col in original_poi_attributes:
    if orig_col in points_with_features_gdf.columns and orig_col not in export_cols:
        export_cols.append(orig_col)
# Add feature columns ensuring they exist
for fc in feature_columns_to_fill:
    if fc in points_with_features_gdf.columns and fc not in export_cols:
        export_cols.append(fc)
if 'is_misplaced' in points_with_features_gdf.columns and 'is_misplaced' not in export_cols:
    export_cols.append('is_misplaced')
if 'geometry' not in export_cols:  # Ensure geometry is in the list
    export_cols.append('geometry')

# Ensure unique columns and geometry is last
actual_export_cols = list(dict.fromkeys([c for c in export_cols if c in points_with_features_gdf.columns]))
if 'geometry' in actual_export_cols:
    actual_export_cols.remove('geometry')
    actual_export_cols.append('geometry')
else:  # If original geometry was dropped somehow, re-add it if possible
    if points_with_features_gdf.geometry.name not in actual_export_cols:
        actual_export_cols.append(points_with_features_gdf.geometry.name)

labeling_export_gdf = points_with_features_gdf[actual_export_cols].copy()
if labeling_export_gdf.geometry.name != 'geometry':
    labeling_export_gdf = labeling_export_gdf.rename_geometry('geometry')

try:
    labeling_export_gdf.to_file(OUTPUT_LABELING_GPKG_PATH, layer="review_points", driver="GPKG")
    print(f"Exported {len(labeling_export_gdf)} points to '{OUTPUT_LABELING_GPKG_PATH}' layer 'review_points'.")
except Exception as e:
    print(f"Error exporting for labeling: {e}")

# --- LOAD MANUAL LABELS ---
print("\n--- Loading Manual Labels (if available) ---")
y_target_column = 'is_misplaced'
final_training_data_gdf = points_with_features_gdf.copy()

try:
    if not os.path.exists(OUTPUT_LABELING_GPKG_PATH):
        print(f"Manual labels file '{OUTPUT_LABELING_GPKG_PATH}' does not exist. Using heuristic labels.")
    else:
        manual_labels_gdf = gpd.read_file(OUTPUT_LABELING_GPKG_PATH, layer="review_points")
        print(f"Read '{OUTPUT_LABELING_GPKG_PATH}' for manual labels.")

        if 'manual_label' in manual_labels_gdf.columns and 'point_id' in manual_labels_gdf.columns:
            print("Found 'manual_label' column in the labeling file.")
            labels_to_merge = manual_labels_gdf[['point_id', 'manual_label']].copy()
            labels_to_merge.dropna(subset=['manual_label'], inplace=True)

            if not labels_to_merge.empty:
                labels_to_merge['manual_label'] = labels_to_merge['manual_label'].astype(int)
                labels_to_merge.drop_duplicates(subset=['point_id'], keep='first', inplace=True)

                current_pwf_point_id_dtype = points_with_features_gdf['point_id'].dtype
                # Ensure 'point_id' is string for merging, if it's not already compatible.
                # This assumes point_id can be safely converted to string and back.
                try:
                    points_with_features_gdf['point_id'] = points_with_features_gdf['point_id'].astype(str)
                    labels_to_merge['point_id'] = labels_to_merge['point_id'].astype(str)
                except Exception as e_conv:
                    print(f"Warning: Could not convert point_id to string for merge: {e_conv}")

                if 'manual_label' in points_with_features_gdf.columns:
                    points_with_features_gdf = points_with_features_gdf.drop(columns=['manual_label'])

                points_with_features_gdf = points_with_features_gdf.merge(
                    labels_to_merge, on='point_id', how='left'
                )
                try:
                    points_with_features_gdf['point_id'] = points_with_features_gdf['point_id'].astype(
                        current_pwf_point_id_dtype)
                except:
                    print("Note: point_id kept as its current type after merge.")

                print(f"Merged manual labels. Value counts for 'manual_label' column after merge:")
                print(points_with_features_gdf['manual_label'].value_counts(dropna=False))

                temp_final_training_gdf = points_with_features_gdf[
                    points_with_features_gdf['manual_label'].isin([0, 1])].copy()
                if not temp_final_training_gdf.empty:
                    final_training_data_gdf = temp_final_training_gdf
                    y_target_column = 'manual_label'
                    print(f"Using '{y_target_column}' for model training. {len(final_training_data_gdf)} samples.")
                else:
                    print("No valid manual labels (0 or 1) found after merge. Using heuristic labels for training.")
            else:
                print("'manual_label' column was present but empty/all NaN. Using heuristic labels for training.")
        else:
            print(
                "Warning: 'point_id' or 'manual_label' not found in labels file. Using heuristic labels for training.")
except Exception as e:
    print(f"Error loading manual labels: {e}. Using heuristic labels for training.")
    traceback.print_exc()

if y_target_column == 'is_misplaced':
    print("Final decision: Using heuristic 'is_misplaced' as the target for training.")
    final_training_data_gdf = points_with_features_gdf.copy()

# --- Prepare for Model Training ---
print("\n--- Preparing Data for Model Training ---")
model_feature_cols = [  # Defined this list here for clarity
    'dist_to_nearest_road', 'dist_to_projected_on_road', 'dist_to_nearest_building',
    'is_on_road_buffer', 'nearby_points_density'
]
actual_model_feature_cols = [col for col in model_feature_cols if col in final_training_data_gdf.columns]
if not actual_model_feature_cols:
    print("Error: No model feature columns available in final_training_data_gdf. Exiting.")
    exit()
if y_target_column not in final_training_data_gdf.columns or final_training_data_gdf[y_target_column].isnull().all():
    print(f"Error: Target column '{y_target_column}' is missing or all NaN in final_training_data_gdf. Exiting.")
    exit()

X = final_training_data_gdf[actual_model_feature_cols].copy()
y = final_training_data_gdf[y_target_column].copy()
print(f"Training with X shape: {X.shape}, y shape: {y.shape}, target: '{y_target_column}'")

for col in X.columns:
    if X[col].isnull().all():
        X[col] = X[col].fillna(0)
    elif X[col].isnull().any():
        X[col] = X[col].fillna(X[col].mean())

if y.nunique() < 2 or len(X) < 10:
    print(f"Warning: Insufficient data or classes for target '{y_target_column}'. Skipping model training.")
    points_with_features_gdf['detected_misplaced_proba'] = 0.0
    # Default detected_misplaced based on the target column (could be heuristic or sparse manual)
    target_col_for_default_pred = final_training_data_gdf[
        y_target_column] if y_target_column in final_training_data_gdf else points_with_features_gdf['is_misplaced']
    points_with_features_gdf['detected_misplaced'] = target_col_for_default_pred.mode()[
        0] if not target_col_for_default_pred.empty and not target_col_for_default_pred.mode().empty else 0

else:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training detection model with {len(X_train)} samples.")
    detection_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    detection_model.fit(X_train, y_train)

    y_pred_detection_test = detection_model.predict(X_test)
    print("\nDetection Model Performance (on test set):")
    print(classification_report(y_test, y_pred_detection_test, zero_division=0))

    model_classes = sorted(y.unique())  # Use classes present in y for labels
    cm = confusion_matrix(y_test, y_pred_detection_test, labels=model_classes)
    print("Confusion Matrix (test set):\n", cm)
    try:
        plt.figure(figsize=(6, 4));
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model_classes, yticklabels=model_classes)
        plt.xlabel('Predicted');
        plt.ylabel('Actual');
        plt.title('Confusion Matrix (Test Set)');
        plt.savefig("confusion_matrix.png");
        plt.close()
        print("Confusion matrix saved to confusion_matrix.png")
    except Exception as plot_e:
        print(f"Could not plot/save confusion matrix: {plot_e}")

    X_full_dataset = points_with_features_gdf[actual_model_feature_cols].copy()
    for col in X_full_dataset.columns:
        if X_full_dataset[col].isnull().all():
            X_full_dataset[col] = X_full_dataset[col].fillna(0)
        elif X_full_dataset[col].isnull().any():
            X_full_dataset[col] = X_full_dataset[col].fillna(X_full_dataset[col].mean())

    points_with_features_gdf['detected_misplaced_proba'] = detection_model.predict_proba(X_full_dataset)[:, 1]
    misplacement_threshold = 0.5
    points_with_features_gdf['detected_misplaced'] = (
                points_with_features_gdf['detected_misplaced_proba'] > misplacement_threshold).astype(int)
    print(
        f"\nTotal points in full dataset detected as misplaced by model (threshold {misplacement_threshold}): {points_with_features_gdf['detected_misplaced'].sum()}")
    print("Value counts for 'detected_misplaced' (full dataset):\n",
          points_with_features_gdf['detected_misplaced'].value_counts(dropna=False))

print("\nScript finished.")