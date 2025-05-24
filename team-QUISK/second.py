"""
Comprehensive solution for automatic roundabout identification
using probe data and machine learning techniques.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import time
import math
import requests
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, pairwise_distances
from sklearn.utils import class_weight
from scipy.spatial import KDTree
import hdbscan
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import BatchNormalization
from kneed import KneeLocator
import matplotlib.patches as patches
import random

# ======== Utility Functions ========

def xy(lnglat, truncate=False):
    """Convert longitude and latitude to web mercator x, y"""
    if isinstance(lnglat, pd.DataFrame):
        lng, lat = lnglat['longitude'].values, lnglat['latitude'].values
    else:
        lng, lat = lnglat[:, 0], lnglat[:, 1]
    
    if truncate:
        lng = np.clip(lng, -180.0, 180.0)
        lat = np.clip(lat, -90.0, 90.0)
    
    x = 6378137.0 * np.radians(lng)
    y = 6378137.0 * np.log(np.tan((math.pi * 0.25) + (0.5 * np.radians(lat))))
    
    if isinstance(lnglat, pd.DataFrame):
        return pd.DataFrame({'x': x, 'y': y})
    return np.array((x, y)).T

def latlng_to_tile(lng, lat, zoom):
    """Convert latitude and longitude to tile coordinates"""
    n = 2 ** zoom
    lat_rad = np.radians(lat)
    tile_x = int((lng + 180.0) / 360.0 * n)
    tile_y = int((1.0 - np.log(np.tan(lat_rad) + (1 / np.cos(lat_rad))) / math.pi) / 2.0 * n)
    return tile_x, tile_y

def calculate_heading_change(group):
    """Calculate mean absolute heading change for a group"""
    return np.abs(np.diff(group['heading'])).mean() if len(group) > 1 else 0

def calculate_curvature(group):
    """Calculate mean curvature for a group"""
    if len(group) < 3:
        return 0
    x = group['longitude'].values
    y = group['latitude'].values
    dx = np.gradient(x)
    dy = np.gradient(y)
    ds = np.sqrt(dx*dx + dy*dy)
    d2x = np.gradient(dx, ds, edge_order=1)
    d2y = np.gradient(dy, ds, edge_order=1)
    curvature = np.abs(dx * d2y - dy * d2x) / (dx * dx + dy * dy)**1.5
    return np.mean(curvature)

def analyze_heading(cluster_df):
    """Analyze heading changes within a cluster"""
    heading_changes = np.diff(cluster_df['heading'].values)
    return np.mean(heading_changes)

def extract_cluster_features(cluster_df):
    """Extract comprehensive features from a cluster"""
    features = {
        'mean_heading_change': cluster_df['heading_change'].mean(),
        'std_heading_change': cluster_df['heading_change'].std(),
        'mean_speed': cluster_df['speed'].mean(),
        'std_speed': cluster_df['speed'].std(),
        'cluster_size': len(cluster_df),
        'latitude': cluster_df['latitude'].mean(),
        'longitude': cluster_df['longitude'].mean(),
        'density': len(cluster_df) / (np.max(cluster_df['longitude']) - np.min(cluster_df['longitude'])) / 
                  (np.max(cluster_df['latitude']) - np.min(cluster_df['latitude'])) if len(cluster_df) > 1 else 0,
        'radius': np.mean(np.sqrt((cluster_df['longitude'] - cluster_df['longitude'].mean())**2 + 
                               (cluster_df['latitude'] - cluster_df['latitude'].mean())**2))
    }
    return features

def process_file(fname):
    """Process a single probe data file"""
    try:
        print(f"Processing {os.path.basename(fname)}")
        df = pd.read_csv(fname)
        coords_list = []
        df['sampledate'] = pd.to_datetime(df['sampledate'])
        for _, group in df.groupby(['traceid']):
            try:
                group = group[group['speed'] != 0].sort_values('sampledate')
                if len(group) < 2 or group['speed'].max() < 15:
                    continue
                time_diff = group['sampledate'].diff().dt.total_seconds()
                heading_diff = group['heading'].diff()
                derivative = heading_diff / time_diff
                filtered_group = group[derivative < -20]  # Adjust threshold as needed
                coords_list.extend(filtered_group[['latitude', 'longitude']].values.tolist())
            except Exception as e:
                print(f"Error processing group in {os.path.basename(fname)}: {e}")
                continue
        return coords_list
    except Exception as e:
        print(f"Error processing file {os.path.basename(fname)}: {e}")
        return []  # Return empty list on error

# New helper function for safe parallel processing
def safe_process_files(files, process_func, max_workers=None):
    """Safely process files in parallel with fallback to sequential processing"""
    if max_workers is None:
        # Use half the CPU cores to avoid memory issues
        max_workers = max(1, os.cpu_count() // 2)
    
    try:
        print(f"Attempting parallel processing with {max_workers} workers")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Use timeout to avoid hanging workers
            results = list(executor.map(process_func, files, timeout=300))
        return results
    except (BrokenProcessPool, TimeoutError, Exception) as e:
        print(f"Parallel processing failed: {e}")
        print("Falling back to sequential processing...")
        results = []
        for file in files:
            result = process_func(file)
            results.append(result)
        return results

# ======== Trajectory-based Roundabout Detection ========

def trajectory_based_detection(data_dir, known_roundabouts_path, num_files=10):
    """Detect roundabouts using trajectory-based approach"""
    print("\n=== Starting Trajectory-Based Detection ===")
    
    # Load data
    start_time = time.time()
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if num_files > 0:
        csv_files = csv_files[:num_files]  # Limit the number of files for testing
        
    print(f"Loading {len(csv_files)} probe data files...")
    
    # Process files using the safe parallel processing function
    results = safe_process_files(csv_files, process_file)
    
    # Flatten results into a single list of coordinates
    coords_list = [coord for sublist in results for coord in sublist if sublist]  # Skip empty results
    
    if not coords_list:
        print("No valid coordinates found. Check your data sources.")
        return None, None
    
    coords = np.array(coords_list)
    
    print(f"Processed {len(coords)} data points in {time.time() - start_time:.2f} seconds")
    
    # Load known roundabouts for validation
    roundabout_data = pd.read_csv(known_roundabouts_path)
    print(f"Loaded {len(roundabout_data)} known roundabouts")
    
    # Prepare data for clustering
    features = pd.DataFrame(coords, columns=['latitude', 'longitude'])
    
    # First-level clustering to group nearby points
    print("Performing DBSCAN clustering...")
    epsilon = 0.0005  # Adjust based on data scale
    min_samples = 5
    db = DBSCAN(eps=epsilon, min_samples=min_samples)
    features['cluster'] = db.fit_predict(coords)
    
    print(f"Found {len(set(features[features['cluster'] != -1]['cluster']))} clusters")
    
    # Extract cluster features
    clustered_points = features[features['cluster'] != -1]
    cluster_features_list = []
    
    for cluster_id, cluster_df in clustered_points.groupby('cluster'):
        # Skip clusters with too few points
        if len(cluster_df) < 10:
            continue
            
        # Calculate cluster centroid
        centroid_lat = cluster_df['latitude'].mean()
        centroid_lng = cluster_df['longitude'].mean()
        
        # Calculate distance of each point to centroid
        cluster_df['distance_to_center'] = np.sqrt(
            (cluster_df['latitude'] - centroid_lat)**2 +
            (cluster_df['longitude'] - centroid_lng)**2
        )
        
        # Extract features for this cluster
        cluster_features = {
            'cluster_id': cluster_id,
            'latitude': centroid_lat,
            'longitude': centroid_lng,
            'point_count': len(cluster_df),
            'mean_distance': cluster_df['distance_to_center'].mean(),
            'std_distance': cluster_df['distance_to_center'].std(),
            'max_distance': cluster_df['distance_to_center'].max(),
            'circularity': cluster_df['distance_to_center'].std() / cluster_df['distance_to_center'].mean()
        }
        
        # Label as roundabout if close to a known roundabout (for training)
        min_distance = float('inf')
        for _, roundabout in roundabout_data.iterrows():
            dist = np.sqrt((centroid_lat - roundabout['latitude'])**2 + 
                          (centroid_lng - roundabout['longitude'])**2)
            if dist < min_distance:
                min_distance = dist
                
        cluster_features['is_roundabout'] = 1 if min_distance < 0.001 else 0  # Threshold for matching
        cluster_features_list.append(cluster_features)
    
    # Convert to DataFrame
    cluster_features_df = pd.DataFrame(cluster_features_list)
    
    if len(cluster_features_df) == 0:
        print("No valid clusters found for analysis")
        return None, None
        
    print(f"Extracted features for {len(cluster_features_df)} clusters")
    
    # Train a machine learning model to identify roundabouts
    X = cluster_features_df.drop(['cluster_id', 'latitude', 'longitude', 'is_roundabout'], axis=1)
    y = cluster_features_df['is_roundabout']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns, 
        'importance': rf_model.feature_importances_
    })
    print("\nFeature Importance:")
    print(feature_importance.sort_values('importance', ascending=False))
    
    # Predict on all clusters
    cluster_features_df['predicted_roundabout'] = rf_model.predict(X)
    
    # Display results
    detected_roundabouts = cluster_features_df[cluster_features_df['predicted_roundabout'] == 1]
    print(f"\nDetected {len(detected_roundabouts)} potential roundabouts")
    
    return cluster_features_df, features

# ======== Advanced Trajectory Analysis ========

def advanced_trajectory_analysis(data_dir, known_roundabouts_path, num_files=10):
    """Perform more advanced trajectory analysis using HDBSCAN"""
    print("\n=== Starting Advanced Trajectory Analysis ===")
    
    # Load data
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if num_files > 0:
        csv_files = csv_files[:num_files]  # Limit the number of files for testing
    
    print(f"Loading {len(csv_files)} probe data files...")
    
    # Load data safely (one file at a time to avoid memory issues)
    dataframes = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dataframes.append(df)
        except Exception as e:
            print(f"Error loading file {os.path.basename(f)}: {e}")
    
    if not dataframes:
        print("No valid data loaded.")
        return None
        
    probe_data = pd.concat(dataframes, ignore_index=True)
    
    # Load known roundabouts
    roundabout_data = pd.read_csv(known_roundabouts_path)
    print(f"Loaded {len(roundabout_data)} known roundabouts")
    
    print(f"Initial probe data shape: {probe_data.shape}")
    
    # Basic preprocessing
    probe_data.dropna(inplace=True)
    probe_data = probe_data[probe_data['speed'] > 0]
    
    # Filter out pedestrians
    max_speeds = probe_data.groupby('traceid')['speed'].max()
    pedestrians = max_speeds[max_speeds < 15].index
    probe_data = probe_data[~probe_data['traceid'].isin(pedestrians)]
    
    print(f"Probe data shape after filtering: {probe_data.shape}")
    
    # Feature engineering
    probe_data['heading_change'] = probe_data.groupby('traceid')['heading'].diff().fillna(0).abs()
    
    # Group by traceid and calculate features
    grouped = probe_data.groupby('traceid')
    features = grouped.agg({
        'latitude': 'mean',
        'longitude': 'mean',
        'speed': 'mean',
    }).reset_index()
    
    features['heading_change'] = grouped.apply(calculate_heading_change)
    features['curvature'] = grouped.apply(calculate_curvature)
    
    # Add additional features
    features['point_count'] = grouped.size()
    features['distance'] = grouped.apply(lambda g: np.sum(np.sqrt(
        np.diff(g['longitude'])**2 + np.diff(g['latitude'])**2)) if len(g) > 1 else 0
    )
    
    # Remove any rows with NaN values
    features = features.dropna()
    print(f"Features shape after processing: {features.shape}")
    
    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features.drop(['traceid', 'latitude', 'longitude'], axis=1))
    
    # Apply HDBSCAN clustering
    if scaled_features.shape[0] > 0:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
        clusterer.fit(scaled_features)
        
        # Add cluster labels to features
        features['cluster'] = clusterer.labels_
        
        # Label data points as roundabouts or not
        def is_roundabout(lat, lon):
            return any((roundabout_data['latitude'] - lat)**2 + 
                      (roundabout_data['longitude'] - lon)**2 < 1e-6)
        
        features['is_roundabout'] = features.apply(
            lambda row: is_roundabout(row['latitude'], row['longitude']), axis=1
        )
        
        # Prepare data for classification
        X = features.drop(['traceid', 'latitude', 'longitude', 'is_roundabout', 'cluster'], axis=1)
        y = features['is_roundabout']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = rf_model.predict(X_test)
        print("\nModel Evaluation:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns, 
            'importance': rf_model.feature_importances_
        })
        print("\nFeature Importance:")
        print(feature_importance.sort_values('importance', ascending=False))
        
        # Predict roundabouts on all data
        features['predicted_roundabout'] = rf_model.predict(X)
        
        print(f"\nDetected {features['predicted_roundabout'].sum()} potential roundabouts")
        
        return features
    else:
        print("No data available after preprocessing.")
        return None

# ======== K-means Clustering with Optimal K Selection ========

def kmeans_detection(data_dir, known_roundabouts_path, batch_id=0, api_key=None):
    """Detect roundabouts using k-means with optimal clustering"""
    print(f"\n=== Starting K-means Detection for Batch {batch_id} ===")
    
    # Setup paths
    path = os.path.join(data_dir, f"{batch_id}/*.csv")
    output_dir = f"centroid_images/{batch_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load roundabouts
    roundabouts = pd.read_csv(known_roundabouts_path)
    roundabout = roundabouts[roundabouts['bbox'] == batch_id]
    
    # Process files using the safe parallel processing function
    results = safe_process_files(glob.glob(path), process_file_for_kmeans)
    
    coords_list = [coord for sublist in results for coord in sublist]
    coords = np.array(coords_list)
    
    if len(coords) == 0:
        print("No coordinates found for analysis")
        return None
    
    # Filter points by distance threshold for density
    filtered_coords = filter_points_by_distance(coords, threshold=0.0005)
    print(f"Filtered {len(coords) - len(filtered_coords)} points, {len(filtered_coords)} remaining")
    
    # Determine optimal number of clusters using elbow method
    distortions = []
    k_range = range(2, 9)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(filtered_coords)
        distortions.append(kmeans.inertia_)
    
    # Use the KneeLocator to find the "elbow" point
    kneedle = KneeLocator(k_range, distortions, curve='convex', direction='decreasing')
    optimal_clusters = kneedle.elbow
    if optimal_clusters is None:
        optimal_clusters = 5  # Default if no clear elbow is found
    else:
        optimal_clusters += 2  # Add a bit more clusters for better granularity
    
    print(f"Optimal number of clusters: {optimal_clusters}")
    
    # Perform k-means clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=420, n_init=10)
    cluster_labels = kmeans.fit_predict(filtered_coords)
    
    # Create DataFrame with results
    filtered_data = pd.DataFrame(filtered_coords, columns=['latitude', 'longitude'])
    filtered_data['cluster'] = cluster_labels
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(filtered_data['longitude'], filtered_data['latitude'], 
                       c=filtered_data['cluster'], cmap='viridis', 
                       marker='o', s=10, alpha=0.6)
    
    ax.scatter(roundabout['longitude'], roundabout['latitude'], 
               c='red', marker='x', s=50, label='Known Roundabouts')
    
    # Get centroids
    centroids = kmeans.cluster_centers_
    
    # If API key is provided, fetch map tiles for centroids
    if api_key:
        for i, centroid in enumerate(centroids):
            lat, lon = centroid[0], centroid[1]
            image_data = fetch_map_image(lon, lat, api_key)
            
            if image_data:
                # Save the image
                image_filename = os.path.join(output_dir, f"map_image_{lat},{lon}.png")
                with open(image_filename, "wb") as image_file:
                    image_file.write(image_data)
                print(f"Saved image for centroid {i + 1} at coordinates ({lat}, {lon})")
    
    # Add squares and centroids to plot
    plot_cluster_squares_and_centroids(ax, filtered_data, cluster_labels)
    
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Clustered Probe Data {batch_id} with {optimal_clusters} clusters')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{batch_id}_cluster_graph.png")
    
    return filtered_data, centroids

# ======== Image-based Roundabout Detection ========

def create_advanced_roundabout_cnn(input_shape=(224, 224, 3), num_classes=1):
    """Create an advanced CNN model for roundabout detection from satellite imagery"""
    model = tf.keras.Sequential([
        # First Convolutional Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        # Second Convolutional Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        # Third Convolutional Block
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.25),
        
        # Fully connected layers
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='sigmoid')
    ])
    
    return model

def train_roundabout_cnn(train_dir, test_dir, epochs=5):
    """Train a CNN model for roundabout detection"""
    # Constants
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_TEST = 20
    
    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=360,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    test_datagen = ImageDataGenerator(rescale=1.0/255)
    
    # Data Generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE_TRAIN,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE_TEST,
        class_mode='binary',
        subset='validation',
        shuffle=True
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE_TEST,
        class_mode='binary',
        shuffle=False
    )
    
    # Calculate class weights
    labels = np.array(train_generator.labels)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weight_dict)
    
    # Create and compile model
    model = create_advanced_roundabout_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(threshold=0.3),
            tf.keras.metrics.Precision(thresholds=0.3),
            tf.keras.metrics.Recall(thresholds=0.3)
        ]
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001
    )
    
    # Train model
    print("Training CNN model...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight_dict
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy, test_precision, test_recall = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test precision: {test_precision:.4f}")
    print(f"Test recall: {test_recall:.4f}")
    
    # Predict with custom threshold
    predictions = model.predict(test_generator)
    y_pred = (predictions > 0.3).astype(int)
    y_true = test_generator.labels
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    print(f"Custom threshold accuracy: {accuracy:.4f}")
    print(f"Custom threshold precision: {precision:.4f}")
    print(f"Custom threshold recall: {recall:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
    
    return model, history

def predict_with_threshold(model, image_path, threshold=0.3):
    """Make prediction with custom threshold"""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0][0]
    result = "roundabout" if prediction > threshold else "no roundabout"
    
    return prediction, result

def fetch_map_tiles(coordinates, api_key, zoom_level=15, output_dir="map_tiles"):
    """Fetch map tiles from HERE API for image-based detection"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_paths = []
    for i, (lng, lat) in enumerate(coordinates):
        # Get tile column and row coordinates
        tile_x, tile_y = latlng_to_tile(lng, lat, zoom_level)
        
        # Prepare the URL for the API request
        url = f"https://maps.hereapi.com/v3/base/mc/{zoom_level}/{tile_x}/{tile_y}/png?apiKey={api_key}"
        
        # Call the API to get the tile image
        response = requests.get(url)
        
        if response.status_code == 200:
            file_path = os.path.join(output_dir, f'tile_{tile_x}_{tile_y}.png')
            
            # Save the image to the specified folder
            with open(file_path, 'wb') as f:
                f.write(response.content)
            print(f"Image {i+1}/{len(coordinates)} saved at {file_path}")
            image_paths.append(file_path)
        else:
            print(f"Failed to fetch tile for {lng}, {lat}. Status code: {response.status_code}")
    
    return image_paths

# ======== Visualization Functions ========

def visualize_clusters(features_df, probe_points=None, roundabout_data=None):
    """Visualize clustering results"""
    plt.figure(figsize=(12, 8))
    
    # Plot all probe points if provided
    if probe_points is not None:
        plt.scatter(probe_points['longitude'], probe_points['latitude'], 
                   c='lightgray', s=1, alpha=0.3, label='Probe Data')
    
    # Plot detected roundabouts
    detected = features_df[features_df['predicted_roundabout'] == 1]
    plt.scatter(detected['longitude'], detected['latitude'], 
               c='red', s=50, marker='o', label='Detected Roundabouts')
    
    # Plot known roundabouts if provided
    if roundabout_data is not None:
        plt.scatter(roundabout_data['longitude'], roundabout_data['latitude'], 
                   c='blue', s=30, marker='x', label='Known Roundabouts')
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Roundabout Detection Results')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('roundabout_detection_results.png')
    plt.show()

# ======== Main Function ========

def main():
    """Main function to run the roundabout detection pipeline"""
    print("======== Roundabout Detection Pipeline ========")
    
    # Setup paths
    data_dir = "data/probe_data"
    roundabout_csv = "data/roundabouts.csv"
    
    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found")
        return
    
    if not os.path.exists(roundabout_csv):
        print(f"Error: Roundabout data '{roundabout_csv}' not found")
        return
    
    # Load known roundabouts for validation
    roundabout_data = pd.read_csv(roundabout_csv)
    
    # Method 1: Trajectory-based detection
    cluster_features, probe_points = trajectory_based_detection(os.path.join(data_dir, "0"), roundabout_csv, num_files=5)
    
    # Method 2: Advanced trajectory analysis
    advanced_features = advanced_trajectory_analysis(os.path.join(data_dir, "0"), roundabout_csv, num_files=5)
    
    # Method 3: K-means with optimal k selection
    # Replace with your actual HERE API key
    api_key = "YOUR_HERE_API_KEY"
    
    # Process batches of data
    batch_results = []
    for batch_id in range(3):  # Process first 3 batches for testing
        kmeans_results, centroids = kmeans_detection(data_dir, roundabout_csv, batch_id=batch_id, api_key=api_key)
        if kmeans_results is not None:
            batch_results.append((kmeans_results, centroids, batch_id))
    
    # Visualize results
    if cluster_features is not None:
        visualize_clusters(cluster_features, probe_points, roundabout_data)
    
    if advanced_features is not None:
        plt.figure(figsize=(12, 8))
        detected = advanced_features[advanced_features['predicted_roundabout'] == 1]
        plt.scatter(advanced_features['longitude'], advanced_features['latitude'], 
                   c='lightgray', s=1, alpha=0.3, label='Trajectories')
        plt.scatter(detected['longitude'], detected['latitude'], 
                   c='red', s=50, marker='o', label='Detected Roundabouts')
        plt.scatter(roundabout_data['longitude'], roundabout_data['latitude'], 
                   c='blue', s=30, marker='x', label='Known Roundabouts')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Advanced Roundabout Detection Results')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('advanced_roundabout_detection_results.png')
        plt.show()
    
    print("\n======== Roundabout Detection Complete ========")
    
    # For image-based detection 
    # Uncomment and set paths to your dataset
    """
    # Method 4: Image-based CNN detection
    print("\n=== Image-Based Detection ===")
    
    # Set paths to your dataset
    train_dir = "data_chicago_hackathon_2024/cnn_model/datasets/train"
    test_dir = "data_chicago_hackathon_2024/cnn_model/datasets/val"
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        # Train the CNN model
        cnn_model, history = train_roundabout_cnn(train_dir, test_dir, epochs=5)
        
        # Save the model
        cnn_model.save('roundabout_detection_model.h5')
        
        # Example prediction on a sample image
        if cluster_features is not None:
            detected = cluster_features[cluster_features['predicted_roundabout'] == 1]
            if len(detected) > 0:
                coordinates = list(zip(detected['longitude'], detected['latitude']))
                image_paths = fetch_map_tiles(coordinates[:5], api_key, zoom_level=15)
                
                for image_path in image_paths:
                    prediction, result = predict_with_threshold(cnn_model, image_path)
                    print(f"Image: {image_path}")
                    print(f"Prediction: {prediction:.4f}, Result: {result}")
    else:
        print("Image dataset not found. Skipping CNN training.")
    """

if __name__ == "__main__":
    main()
