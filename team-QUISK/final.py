import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString
from shapely.ops import unary_union, polygonize
import json
import warnings
from collections import defaultdict
import rtree
import time
import sys
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
warnings.filterwarnings('ignore')

print("=== ADAPTIVE ROUNDABOUT DETECTION ===")

# ADAPTIVE SETTINGS - Data-driven detection
SAMPLE_SIZE = None          
HIGH_QUALITY_MODE = True    
MAX_RESULTS = 20            # Show more results for analysis
CONFIDENCE_THRESHOLD = 0.5  # More reasonable baseline
MAX_CANDIDATES = 200        # Allow more candidates for processing
USE_ST_NAME_HINTS = True    

# Performance settings
MAX_NEARBY_STREETS = 30     
SPATIAL_SAMPLE = 15000      # Increased for better coverage
METHOD_TIMEOUT = 1200       
GRID_SIZE = 50              
MAX_PROCESS_CANDIDATES = 100

# ADAPTIVE validation parameters (will be adjusted based on data)
MIN_ROUNDABOUT_RADIUS = 15  # More permissive minimum
MAX_ROUNDABOUT_RADIUS = 100 # More permissive maximum
MIN_APPROACH_STREETS = 2    # Reduced minimum - some roundabouts may have fewer approaches
MIN_CIRCULARITY = 0.5       # More permissive circularity
MIN_STREET_ANGLE_SEPARATION = 30  # Reduced minimum angle

# Performance monitoring
detection_times = []
total_candidates = 0
valid_roundabouts = 0

# Strict validation parameters
STRICT_MIN_ROUNDABOUT_RADIUS = 20  # Minimum realistic roundabout size
STRICT_MAX_ROUNDABOUT_RADIUS = 60  # Maximum realistic roundabout size
STRICT_MIN_APPROACH_STREETS = 3    # Minimum number of streets approaching roundabout
STRICT_MIN_CIRCULARITY = 0.7       # Minimum circularity for valid roundabout
STRICT_MIN_STREET_ANGLE_SEPARATION = 45  # Minimum angle between approach streets (degrees)

# Add geometric validation helpers
def calculate_circularity(geometry):
    """Calculate circularity metric (4πA/P²) for a geometry."""
    area = geometry.area
    perimeter = geometry.length
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        return min(1.0, circularity)  # Cap at 1.0 to handle numerical precision issues
    return 0.0

def minimum_enclosing_circle(points):
    """Find the minimum enclosing circle for a set of points."""
    try:
        # Convert to numpy array
        coords = np.array([[p.x, p.y] for p in points])
        
        if len(coords) < 3:
            # Not enough points for a meaningful circle
            if len(coords) == 0:
                return None, None
            return Point(np.mean(coords[:, 0]), np.mean(coords[:, 1])), 0
            
        # Calculate centroid and average distance as a fallback
        centroid_x = np.mean(coords[:, 0])
        centroid_y = np.mean(coords[:, 1])
        
        # Calculate distances from centroid
        distances = np.sqrt(((coords[:, 0] - centroid_x) ** 2) + 
                           ((coords[:, 1] - centroid_y) ** 2))
        radius = np.max(distances) * 1.05  # Add 5% margin
        
        return Point(centroid_x, centroid_y), radius
        
    except Exception as e:
        print(f"Error in minimum_enclosing_circle: {e}")
        return None, None

def validate_street_approaches(center, radius, streets_gdf):
    """Validate that streets properly approach the roundabout from multiple directions."""
    buffer = center.buffer(radius * 2.0)  # Larger buffer to find approach streets
    
    # Find streets that intersect the buffer
    potential_streets = []
    for idx, row in streets_gdf.iterrows():
        if buffer.intersects(row.geometry):
            potential_streets.append((idx, row.geometry))
    
    if len(potential_streets) < MIN_APPROACH_STREETS:
        return False, []
    
    # Analyze approach angles
    approach_angles = []
    valid_approaches = []
    
    for idx, street_geom in potential_streets[:15]:  # Limit analysis
        if street_geom.geom_type != 'LineString':
            continue
            
        # Find the point on the street closest to roundabout center
        try:
            closest_point = street_geom.interpolate(street_geom.project(center))
            
            # Check if this point is reasonably close to the roundabout
            distance_to_center = closest_point.distance(center)
            if distance_to_center > radius * 1.8:  # Too far from roundabout
                continue
                
            # Get the direction of the street at the closest point
            coords = list(street_geom.coords)
            
            # Find the coordinate closest to our closest_point
            min_dist = float('inf')
            closest_coord_idx = 0
            for i, coord in enumerate(coords):
                dist = Point(coord).distance(closest_point)
                if dist < min_dist:
                    min_dist = dist
                    closest_coord_idx = i
            
            # Calculate approach angle using adjacent points
            if 0 < closest_coord_idx < len(coords) - 1:
                prev_point = Point(coords[closest_coord_idx - 1])
                next_point = Point(coords[closest_coord_idx + 1])
                
                # Calculate angle towards center
                dx1 = center.x - prev_point.x
                dy1 = center.y - prev_point.y
                dx2 = center.x - next_point.x
                dy2 = center.y - next_point.y
                
                angle1 = np.arctan2(dy1, dx1) * 180 / np.pi
                angle2 = np.arctan2(dy2, dx2) * 180 / np.pi
                
                # Use the angle that points more towards the center
                avg_angle = (angle1 + angle2) / 2
                approach_angles.append(avg_angle)
                valid_approaches.append(idx)
                
        except Exception:
            continue
    
    if len(approach_angles) < MIN_APPROACH_STREETS:
        return False, []
    
    # Check if approach angles are well distributed
    approach_angles.sort()
    angle_separations = []
    
    for i in range(len(approach_angles)):
        next_i = (i + 1) % len(approach_angles)
        separation = approach_angles[next_i] - approach_angles[i]
        if separation < 0:
            separation += 360
        angle_separations.append(separation)
    
    # Check if we have reasonable angle distribution
    min_separation = min(angle_separations)
    if min_separation < MIN_STREET_ANGLE_SEPARATION:
        return False, []
    
    return True, valid_approaches

def is_likely_residential_area(center, radius, streets_gdf):
    """Check if the detected roundabout is in a residential area with inappropriate geometry."""
    buffer = center.buffer(radius * 3)
    
    # Count very short street segments (likely driveways or residential streets)
    short_segments = 0
    total_segments = 0
    
    for idx, row in streets_gdf.iterrows():
        if buffer.intersects(row.geometry):
            total_segments += 1
            if row.geometry.length < 50:  # Very short segments
                short_segments += 1
    
    if total_segments > 0:
        short_segment_ratio = short_segments / total_segments
        # If more than 60% are short segments, likely residential
        if short_segment_ratio > 0.6:
            return True
    
    return False

def validate_roundabout_geometry_strict(geometry, center, radius, nearby_streets, intersection_points=None):
    """Perform strict geometric validation on a potential roundabout."""
    
    # Basic circularity check
    circularity = calculate_circularity(geometry)
    if circularity < MIN_CIRCULARITY:
        return False, 0.0, {}
    
    # Radius check
    if radius < MIN_ROUNDABOUT_RADIUS or radius > MAX_ROUNDABOUT_RADIUS:
        return False, 0.0, {}
    
    # Street approach validation
    streets_gdf = nearby_streets if hasattr(nearby_streets, 'iterrows') else None
    if streets_gdf is not None:
        has_valid_approaches, approach_streets = validate_street_approaches(center, radius, streets_gdf)
        if not has_valid_approaches:
            return False, 0.0, {}
        
        # Check if it's in a residential area
        if is_likely_residential_area(center, radius, streets_gdf):
            return False, 0.0, {}
    
    # If we have intersection points, validate their distribution
    point_distribution_score = 0.5
    if intersection_points and len(intersection_points) >= 4:
        try:
            coords = np.array([[p.x, p.y] for p in intersection_points])
            center_x, center_y = center.x, center.y
            
            # Calculate distances from center
            distances = np.sqrt(((coords[:, 0] - center_x) ** 2) + 
                               ((coords[:, 1] - center_y) ** 2))
            
            # Check if points are roughly at the same distance from center (ring pattern)
            distance_std = np.std(distances)
            distance_mean = np.mean(distances)
            
            if distance_mean > 0:
                distance_uniformity = 1.0 - min(1.0, distance_std / distance_mean)
                point_distribution_score = distance_uniformity
        except Exception:
            pass
    
    # Calculate overall validation score
    validation_score = (circularity * 0.6) + (point_distribution_score * 0.4)
    
    validation_details = {
        'circularity': circularity,
        'point_distribution': point_distribution_score,
        'radius_valid': MIN_ROUNDABOUT_RADIUS <= radius <= MAX_ROUNDABOUT_RADIUS,
        'has_valid_approaches': has_valid_approaches if 'has_valid_approaches' in locals() else False
    }
    
    return validation_score > 0.7, validation_score, validation_details

def analyze_data_characteristics(streets_gdf, intersections):
    """Analyze the dataset to determine adaptive thresholds"""
    print("\n🔍 ANALYZING DATA CHARACTERISTICS FOR ADAPTIVE THRESHOLDS...")
    
    # Calculate street length statistics
    lengths = streets_gdf.geometry.length
    length_stats = {
        'mean': lengths.mean(),
        'median': lengths.median(),
        'std': lengths.std(),
        'q25': lengths.quantile(0.25),
        'q75': lengths.quantile(0.75)
    }
    
    print(f"Street length analysis:")
    print(f"  Mean: {length_stats['mean']:.1f}m")
    print(f"  Median: {length_stats['median']:.1f}m")
    print(f"  25th percentile: {length_stats['q25']:.1f}m")
    print(f"  75th percentile: {length_stats['q75']:.1f}m")
    
    # Estimate typical roundabout sizes for this dataset
    # Roundabouts are typically 2-10 times the median street length in radius
    estimated_min_radius = max(10, length_stats['median'] * 0.05)
    estimated_max_radius = min(200, length_stats['median'] * 0.5)
    
    # Calculate intersection density
    total_area = (streets_gdf.total_bounds[2] - streets_gdf.total_bounds[0]) * \
                 (streets_gdf.total_bounds[3] - streets_gdf.total_bounds[1])
    intersection_density = len(intersections) / (total_area / 1e6)  # per km²
    
    print(f"Intersection density: {intersection_density:.2f} intersections/km²")
    
    # Adapt clustering parameters based on density
    if intersection_density > 50:  # Dense urban area
        dbscan_eps = 25
        min_samples = 3
    elif intersection_density > 10:  # Suburban area
        dbscan_eps = 35
        min_samples = 3
    else:  # Rural area
        dbscan_eps = 50
        min_samples = 2
    
    adaptive_params = {
        'min_radius': estimated_min_radius,
        'max_radius': estimated_max_radius,
        'dbscan_eps': dbscan_eps,
        'min_samples': min_samples,
        'min_approaches': 2 if intersection_density < 10 else 3,
        'min_circularity': 0.4 if intersection_density < 10 else 0.5
    }
    
    print(f"Adaptive parameters:")
    print(f"  Radius range: {adaptive_params['min_radius']:.1f} - {adaptive_params['max_radius']:.1f}m")
    print(f"  DBSCAN eps: {adaptive_params['dbscan_eps']}m")
    print(f"  Min samples: {adaptive_params['min_samples']}")
    print(f"  Min approaches: {adaptive_params['min_approaches']}")
    print(f"  Min circularity: {adaptive_params['min_circularity']:.2f}")
    
    return adaptive_params

def validate_roundabout_adaptive(geometry, center, radius, nearby_streets, intersection_points=None, adaptive_params=None):
    """Perform adaptive validation based on data characteristics"""
    
    if adaptive_params is None:
        adaptive_params = {
            'min_radius': 15,
            'max_radius': 100,
            'min_approaches': 2,
            'min_circularity': 0.5
        }
    
    # Basic circularity check with adaptive threshold
    circularity = calculate_circularity(geometry)
    if circularity < adaptive_params['min_circularity']:
        return False, 0.0, {'reason': 'circularity_too_low', 'circularity': circularity}
    
    # Adaptive radius check
    if radius < adaptive_params['min_radius'] or radius > adaptive_params['max_radius']:
        return False, 0.0, {'reason': 'radius_out_of_range', 'radius': radius}
    
    # Count nearby streets with more permissive criteria
    street_count = 0
    if hasattr(nearby_streets, '__len__'):
        street_count = len(nearby_streets)
    elif hasattr(nearby_streets, 'iterrows'):
        # Count streets that actually intersect with a buffer around the roundabout
        buffer = center.buffer(radius * 1.8)  # Larger buffer
        for idx, row in nearby_streets.iterrows():
            if buffer.intersects(row.geometry):
                street_count += 1
                if street_count >= 10:  # Don't need to count all
                    break
    
    if street_count < adaptive_params['min_approaches']:
        return False, 0.0, {'reason': 'insufficient_approaches', 'street_count': street_count}
    
    # Calculate validation score
    validation_score = circularity * 0.7 + min(1.0, street_count / 6) * 0.3
    
    validation_details = {
        'circularity': circularity,
        'street_count': street_count,
        'radius': radius,
        'validation_score': validation_score
    }
    
    return validation_score > 0.4, validation_score, validation_details

def load_and_analyze_street_data():
    """Load and perform targeted analysis on street network data"""
    print("Loading complete street network for analysis...")
    
    # Check if the file exists first
    shapefile_path = 'Streets.shp'
    if not os.path.exists(shapefile_path):
        print(f"❌ ERROR: File '{shapefile_path}' not found in current directory")
        print(f"Current directory: {os.getcwd()}")
        print("Available files:")
        for file in os.listdir('.'):
            if file.endswith(('.shp', '.geojson', '.json')):
                print(f"  - {file}")
        
        # Try alternative names
        alternative_names = ['streets.shp', 'street.shp', 'roads.shp', 'road.shp']
        for alt_name in alternative_names:
            if os.path.exists(alt_name):
                print(f"Found alternative file: {alt_name}")
                shapefile_path = alt_name
                break
        else:
            raise FileNotFoundError(f"No street data file found. Please ensure 'Streets.shp' exists in {os.getcwd()}")
    
    try:
        print(f"Attempting to load: {shapefile_path}")
        streets = gpd.read_file(shapefile_path)
        print(f"✓ Successfully loaded {len(streets)} street segments")
    except Exception as e:
        print(f"❌ ERROR loading shapefile: {e}")
        print(f"File size: {os.path.getsize(shapefile_path) if os.path.exists(shapefile_path) else 'N/A'} bytes")
        raise
    
    # Check if the data is empty
    if len(streets) == 0:
        raise ValueError("The loaded street data is empty")
    
    # Analyze street attributes more thoroughly
    print("\n📊 STREET DATA ANALYSIS:")
    
    # Check current CRS
    print(f"Original CRS: {streets.crs}")
    
    # Convert to projected CRS for accurate distance calculations
    try:
        if streets.crs is None:
            print("⚠️ WARNING: No CRS defined, assuming WGS84")
            streets = streets.set_crs("EPSG:4326")
        
        # Convert to projected CRS (Web Mercator for global use)
        if not streets.crs.is_projected:
            print("Converting to projected CRS (EPSG:3857)...")
            streets = streets.to_crs("EPSG:3857")
            print(f"✓ Converted to CRS: {streets.crs}")
        else:
            print(f"✓ Already in projected CRS: {streets.crs}")
            
    except Exception as e:
        print(f"❌ ERROR with CRS conversion: {e}")
        print("Attempting to continue without CRS conversion...")
    
    # Print column information to better understand the data
    print(f"Columns in street data: {list(streets.columns)}")
    
    # Check for null geometries
    null_geoms = streets.geometry.isnull().sum()
    if null_geoms > 0:
        print(f"⚠️ WARNING: {null_geoms} null geometries found, removing them")
        streets = streets[streets.geometry.notnull()]
    
    # Check for empty geometries
    empty_geoms = streets.geometry.is_empty.sum()
    if empty_geoms > 0:
        print(f"⚠️ WARNING: {empty_geoms} empty geometries found, removing them")
        streets = streets[~streets.geometry.is_empty]
    
    print(f"After cleanup: {len(streets)} street segments")
    
    # Analyze street types and properties if available
    street_type_columns = ['ST_NAME', 'LINK_ID', 'OBJECTID', 'NAME', 'STREET_NAME', 'ROAD_NAME']
    
    for col in street_type_columns:
        if col in streets.columns:
            non_null_count = streets[col].notna().sum()
            if non_null_count > 0:
                value_counts = streets[col].value_counts().head(10)
                print(f"\nTop values in {col} ({non_null_count} non-null values):")
                print(value_counts)
            else:
                print(f"\nColumn {col} exists but has no non-null values")
    
    # Analyze the geometry characteristics
    geom_types = streets.geometry.geom_type.value_counts()
    print(f"\nGeometry types: {geom_types.to_dict()}")
    
    # Check if we have LineString geometries (required for street analysis)
    if 'LineString' not in geom_types:
        print("❌ ERROR: No LineString geometries found in the data")
        print("This data may not represent street networks properly")
        raise ValueError("No LineString geometries found - data may not be street network data")
    
    # Calculate length statistics (handle potential CRS issues)
    try:
        lengths = streets.geometry.length
        valid_lengths = lengths[lengths > 0]
        
        if len(valid_lengths) > 0:
            print(f"Street length stats: min={valid_lengths.min():.1f}m, max={valid_lengths.max():.1f}m, mean={valid_lengths.mean():.1f}m")
            
            # Check if lengths seem reasonable (if they're too small, might be in wrong units)
            if valid_lengths.mean() < 1:
                print("⚠️ WARNING: Average street length is very small - check if data is in correct units/projection")
        else:
            print("⚠️ WARNING: No valid street lengths found")
            
    except Exception as e:
        print(f"⚠️ WARNING: Could not calculate street lengths: {e}")
    
    # Sample data if requested
    if SAMPLE_SIZE and len(streets) > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} segments from {len(streets)} total segments")
        streets = streets.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    
    print(f"Final dataset: {len(streets)} segments ready for analysis")
    
    # Create spatial index to verify data is working
    try:
        print("Testing spatial index creation...")
        _ = streets.sindex
        print("✓ Spatial index created successfully")
    except Exception as e:
        print(f"⚠️ WARNING: Could not create spatial index: {e}")
    
    return streets

def check_for_roundabout_names(streets_gdf):
    """Check for street names that might indicate roundabouts"""
    print("\nChecking for roundabout-indicating street names...")
    roundabout_keywords = [
        'roundabout', 'round', 'circle', 'circular', 'rotary', 'rotunda', 
        'aplis', 'riņķis', 'apļveida'  # Latvian terms for roundabout/circle
    ]
    
    possible_roundabouts = []
    
    if 'ST_NAME' in streets_gdf.columns:
        # Check for roundabout-related street names
        for keyword in roundabout_keywords:
            matches = streets_gdf[streets_gdf['ST_NAME'].str.lower().str.contains(keyword, na=False)]
            if not matches.empty:
                print(f"  Found {len(matches)} streets containing '{keyword}'")
                possible_roundabouts.extend(matches.index.tolist())
    
    print(f"  Found {len(possible_roundabouts)} streets with roundabout-indicating names")
    return possible_roundabouts

def build_street_network_graph(streets_gdf):
    """Build a proper street network graph for intersection analysis"""
    start_time = time.time()
    print("\nBuilding street network connectivity graph...")
    
    # Create spatial index for efficient intersection checks
    print("Creating spatial index...")
    spatial_index = rtree.index.Index()
    for idx, geom in enumerate(streets_gdf.geometry):
        spatial_index.insert(idx, geom.bounds)
    
    # Find all intersections between streets
    print("Finding street intersections...")
    intersections = []
    connections = defaultdict(set)  # Track which streets connect to each other
    
    # Use a reasonable sample for intersection analysis if data is very large
    sample_size = min(len(streets_gdf), SPATIAL_SAMPLE)
    streets_sample = streets_gdf.iloc[:sample_size] if sample_size < len(streets_gdf) else streets_gdf
    
    for idx1, row1 in streets_sample.iterrows():
        if idx1 % 1000 == 0:
            print(f"  Processing street {idx1}/{len(streets_sample)}")
        
        geom1 = row1.geometry
        bounds = geom1.bounds
        
        # Find potential intersecting streets using spatial index
        # Limit to reasonable number to prevent memory issues
        potential_matches = list(spatial_index.intersection(bounds))[:1000]
        
        for idx2 in potential_matches:
            if idx1 >= idx2:
                continue  # Avoid self-intersections and duplicates
            
            try:
                geom2 = streets_gdf.iloc[idx2].geometry
                intersection = geom1.intersection(geom2)
                
                if not intersection.is_empty:
                    if intersection.geom_type == 'Point':
                        intersections.append((intersection, idx1, idx2))
                        connections[idx1].add(idx2)
                        connections[idx2].add(idx1)
                    elif intersection.geom_type == 'MultiPoint':
                        for point in intersection.geoms:
                            intersections.append((point, idx1, idx2))
                        connections[idx1].add(idx2)
                        connections[idx2].add(idx1)
            except Exception:
                continue
    
    intersection_points = [i[0] for i in intersections]
    
    print(f"Found {len(intersections)} street intersections")
    
    # Find potential roundabout nodes (intersections with multiple connections)
    print("Analyzing intersection connectivity...")
    intersection_density = defaultdict(int)
    for _, idx1, idx2 in intersections:
        intersection_density[idx1] += 1
        intersection_density[idx2] += 1
    
    # Streets with high intersection counts are likely part of complex junctions or roundabouts
    # Only keep streets connecting to 3+ other streets
    high_connectivity_streets = {idx: count for idx, count in intersection_density.items() 
                                if count >= 3}
    
    print(f"Found {len(high_connectivity_streets)} streets with high connectivity")
    print(f"Network graph built in {time.time() - start_time:.1f} seconds")
    
    return {
        'intersections': intersections[:100000],  # Limit to prevent memory issues
        'intersection_points': intersection_points[:100000],
        'connections': connections,
        'high_connectivity_streets': high_connectivity_streets
    }

def detect_roundabouts_from_network(streets_gdf, network_data):
    """Find roundabouts from street network structure with strict validation"""
    print("\n[METHOD 1] Network-based roundabout detection (STRICT MODE)...")
    start_time = time.time()
    
    candidates = []
    intersections = network_data['intersections']
    
    # Use DBSCAN with more conservative parameters
    use_dbscan = True
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        use_dbscan = False
        print("  sklearn not available, using grid-based clustering")
    
    if use_dbscan and len(network_data['intersection_points']) > 10:
        print("Using DBSCAN for intersection clustering with strict parameters...")
        
        try:
            intersection_points = network_data['intersection_points']
            max_points = min(len(intersection_points), 20000)  # Reduced for performance
            sample_points = intersection_points[:max_points]
            
            coords = np.array([[p.x, p.y] for p in sample_points])
            
            # More conservative DBSCAN parameters
            eps = 35  # Smaller epsilon for tighter clusters
            min_samples = 4  # More points required for a cluster
            
            print(f"Running DBSCAN with eps={eps}m, min_samples={min_samples}...")
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
            
            labels = clustering.labels_
            cluster_labels = set(labels)
            if -1 in cluster_labels:
                cluster_labels.remove(-1)
            
            print(f"Found {len(cluster_labels)} intersection clusters")
            
            # Process clusters with strict validation
            for label in cluster_labels:
                cluster_mask = labels == label
                cluster_points = [sample_points[i] for i, is_in_cluster in enumerate(cluster_mask) if is_in_cluster]
                
                if len(cluster_points) >= 6:  # Require more points for better confidence
                    from shapely.geometry import MultiPoint
                    center, radius = minimum_enclosing_circle(cluster_points)
                    
                    if center and MIN_ROUNDABOUT_RADIUS <= radius <= MAX_ROUNDABOUT_RADIUS:
                        circle = center.buffer(radius)
                        
                        # Strict geometric validation
                        is_valid, validation_score, details = validate_roundabout_geometry_strict(
                            circle, center, radius, streets_gdf, cluster_points)
                        
                        if not is_valid:
                            continue
                        
                        # Count properly intersecting streets
                        buffer = center.buffer(radius * 1.3)
                        nearby_streets = []
                        
                        for idx, row in streets_gdf.iterrows():
                            if buffer.intersects(row.geometry):
                                # Check if street actually approaches the roundabout
                                closest_point = row.geometry.interpolate(row.geometry.project(center))
                                if closest_point.distance(center) <= radius * 1.5:
                                    nearby_streets.append(row)
                                    if len(nearby_streets) >= MAX_NEARBY_STREETS:
                                        break
                        
                        if len(nearby_streets) < MIN_APPROACH_STREETS:
                            continue
                        
                        # Calculate conservative confidence
                        intersection_density = len(cluster_points)
                        street_connectivity = len(nearby_streets)
                        
                        # Strict confidence calculation
                        density_factor = min(1.0, intersection_density / 12)  # Higher threshold
                        connectivity_factor = min(1.0, street_connectivity / 8)  # Higher threshold
                        geometric_factor = validation_score
                        
                        confidence = (0.4 * geometric_factor + 
                                     0.3 * connectivity_factor + 
                                     0.3 * density_factor)
                        
                        # Only accept high-confidence candidates
                        if confidence >= 0.75:  # Much higher threshold
                            candidates.append({
                                'geometry': circle,
                                'radius': radius,
                                'area': circle.area,
                                'circularity': details['circularity'],
                                'perimeter': circle.length,
                                'intersection_density': float(intersection_density),
                                'connectivity': float(street_connectivity),
                                'nearby_streets': len(nearby_streets),
                                'method': 'network_analysis',
                                'confidence': confidence,
                                'quality_score': 0.6 * confidence + 0.4 * geometric_factor,
                                'validation_details': details
                            })
                            
                            print(f"  ✓ HIGH-QUALITY ROUNDABOUT: {radius:.0f}m radius, {len(nearby_streets)} streets, confidence {confidence:.2f}")
                        
                        if len(candidates) >= MAX_CANDIDATES // 3:  # Even fewer candidates
                            break
                            
        except Exception as e:
            print(f"  DBSCAN error: {e}")
    
    print(f"Method 1 completed in {time.time() - start_time:.1f} seconds, found {len(candidates)} high-quality candidates")
    return candidates

def detect_curved_street_roundabouts(streets_gdf, roundabout_name_indices=None):
    """Find roundabouts by analyzing curved streets with strict validation"""
    print("\n[METHOD 2] Curved street analysis (STRICT MODE)...")
    start_time = time.time()
    
    candidates = []
    curved_streets = []
    
    # Only analyze streets with roundabout names or very high curvature
    priority_indices = set(roundabout_name_indices if roundabout_name_indices else [])
    
    print(f"Analyzing curvature for priority streets and high-curvature segments...")
    
    # Analyze all streets but with much stricter criteria
    for idx, row in streets_gdf.iterrows():
        if idx % 10000 == 0:
            print(f"  Processed {idx}/{len(streets_gdf)} streets")
            
        geom = row.geometry
        if geom.geom_type != 'LineString':
            continue
            
        coords = list(geom.coords)
        if len(coords) < 5:  # Need more points for meaningful curvature
            continue
            
        start_point = Point(coords[0])
        end_point = Point(coords[-1])
        straight_dist = start_point.distance(end_point)
        road_dist = geom.length
        
        if road_dist > 0:
            curvature = 1 - (straight_dist / road_dist)
            
            # Much stricter curvature threshold
            is_priority = idx in priority_indices
            curvature_threshold = 0.4 if is_priority else 0.6  # Higher thresholds
            length_min = 80 if is_priority else 120  # Longer segments required
            length_max = 400  # Upper limit
            
            if (curvature > curvature_threshold and 
                length_min < road_dist < length_max):
                
                curved_streets.append({
                    'geometry': geom,
                    'curvature': curvature,
                    'length': road_dist,
                    'idx': idx,
                    'is_priority': is_priority
                })
    
    print(f"Found {len(curved_streets)} highly curved street segments")
    
    if curved_streets:
        # Group curved streets more conservatively
        spatial_index = rtree.index.Index()
        for idx, street in enumerate(curved_streets):
            spatial_index.insert(idx, street['geometry'].bounds)
        
        processed = set()
        curved_groups = []
        
        for idx, street in enumerate(curved_streets):
            if idx in processed or len(curved_groups) >= 20:  # Limit groups
                continue
                
            street_geom = street['geometry']
            potential_matches = list(spatial_index.intersection(street_geom.bounds))[:50]
            
            group = [idx]
            processed.add(idx)
            is_priority = street['is_priority']
            
            for match_idx in potential_matches:
                if match_idx != idx and match_idx not in processed:
                    match_geom = curved_streets[match_idx]['geometry']
                    if street_geom.distance(match_geom) < 30:  # Closer proximity required
                        group.append(match_idx)
                        processed.add(match_idx)
                        if curved_streets[match_idx]['is_priority']:
                            is_priority = True
            
            # Require more curved streets for a valid roundabout
            if len(group) >= 3:  # At least 3 curved segments
                curved_groups.append({
                    'members': [curved_streets[i] for i in group],
                    'is_priority': is_priority
                })
        
        print(f"Found {len(curved_groups)} potential curved street groups")
        
        for group_data in curved_groups:
            group = group_data['members']
            geometries = [street['geometry'] for street in group]
            
            # Calculate center and radius more carefully
            all_points = []
            for geom in geometries:
                coords = list(geom.coords)
                # Sample points along the curve
                for i in range(0, len(coords), max(1, len(coords)//10)):
                    all_points.append(Point(coords[i]))
            
            if len(all_points) >= 8:  # Need sufficient points
                mp = MultiPoint(all_points)
                center, radius = minimum_enclosing_circle(all_points)
                
                if center and MIN_ROUNDABOUT_RADIUS <= radius <= MAX_ROUNDABOUT_RADIUS:
                    roundabout = center.buffer(radius)
                    
                    # Strict validation
                    is_valid, validation_score, details = validate_roundabout_geometry_strict(
                        roundabout, center, radius, streets_gdf, all_points)
                    
                    if is_valid:
                        avg_curvature = sum(street['curvature'] for street in group) / len(group)
                        
                        # Conservative confidence calculation
                        curvature_factor = min(1.0, avg_curvature / 0.7)
                        group_size_factor = min(1.0, len(group) / 6)
                        priority_bonus = 0.15 if group_data['is_priority'] else 0
                        
                        confidence = (0.4 * validation_score + 
                                     0.3 * curvature_factor + 
                                     0.2 * group_size_factor + 
                                     priority_bonus)
                        
                        if confidence >= 0.8:  # Very high threshold
                            candidates.append({
                                'geometry': roundabout,
                                'radius': radius,
                                'area': roundabout.area,
                                'circularity': details['circularity'],
                                'perimeter': roundabout.length,
                                'curved_streets': len(group),
                                'avg_curvature': avg_curvature,
                                'method': 'curved_streets',
                                'confidence': confidence,
                                'quality_score': 0.7 * confidence + 0.3 * validation_score,
                                'validation_details': details
                            })
                            
                            print(f"  ✓ HIGH-QUALITY CURVED ROUNDABOUT: {radius:.0f}m radius, {len(group)} segments, confidence {confidence:.2f}")
    
    print(f"Method 2 completed in {time.time() - start_time:.1f} seconds, found {len(candidates)} candidates")
    return candidates

def detect_junction_patterns(streets_gdf, network_data):
    """Find roundabouts by detecting junction patterns with strict validation"""
    print("\n[METHOD 3] Junction pattern analysis (STRICT MODE)...")
    start_time = time.time()
    
    candidates = []
    intersection_points = network_data['intersection_points']
    
    if len(intersection_points) < 20:
        print("  Insufficient intersection data for junction pattern analysis")
        return candidates
    
    try:
        from sklearn.cluster import DBSCAN
        
        print(f"Clustering {len(intersection_points)} intersection points with strict parameters...")
        
        max_points = min(len(intersection_points), 15000)
        sample_points = intersection_points[:max_points]
        coords = np.array([[p.x, p.y] for p in sample_points])
        
        # Very conservative clustering
        eps = 25  # Smaller radius
        min_samples = 6  # More points required
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        
        labels = clustering.labels_
        cluster_labels = set(labels)
        if -1 in cluster_labels:
            cluster_labels.remove(-1)
            
        print(f"Found {len(cluster_labels)} intersection clusters")
        
        # Only process the largest clusters
        label_counts = {}
        for label in cluster_labels:
            label_counts[label] = np.sum(labels == label)
        
        sorted_clusters = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
        top_clusters = sorted_clusters[:min(10, len(sorted_clusters))]  # Only top 10
        
        for label, count in top_clusters:
            cluster_mask = labels == label
            cluster_points = [sample_points[i] for i, is_in_cluster in enumerate(cluster_mask) if is_in_cluster]
            
            if len(cluster_points) >= 8:  # Require many intersection points
                center, radius = minimum_enclosing_circle(cluster_points)
                
                if center and MIN_ROUNDABOUT_RADIUS <= radius <= MAX_ROUNDABOUT_RADIUS:
                    roundabout = center.buffer(radius)
                    
                    # Strict validation
                    is_valid, validation_score, details = validate_roundabout_geometry_strict(
                        roundabout, center, radius, streets_gdf, cluster_points)
                    
                    if is_valid:
                        # Conservative confidence calculation
                        intersection_factor = min(1.0, len(cluster_points) / 15)  # Higher threshold
                        geometric_factor = validation_score
                        
                        confidence = 0.6 * geometric_factor + 0.4 * intersection_factor
                        
                        if confidence >= 0.8:  # Very high threshold
                            candidates.append({
                                'geometry': roundabout,
                                'radius': radius,
                                'area': roundabout.area,
                                'circularity': details['circularity'],
                                'perimeter': roundabout.length,
                                'intersection_count': len(cluster_points),
                                'method': 'junction_pattern',
                                'confidence': confidence,
                                'quality_score': 0.8 * confidence + 0.2 * geometric_factor,
                                'validation_details': details
                            })
                            
                            print(f"  ✓ HIGH-QUALITY JUNCTION ROUNDABOUT: {radius:.0f}m radius, {len(cluster_points)} intersections, confidence {confidence:.2f}")
        
    except ImportError:
        print("  sklearn not available, skipping junction pattern detection")
    except Exception as e:
        print(f"  Error in junction pattern detection: {e}")
    
    print(f"Method 3 completed in {time.time() - start_time:.1f} seconds, found {len(candidates)} candidates")
    return candidates

def timeout_handler(func, *args, **kwargs):
    """Execute a function with a timeout"""
    start_time = time.time()
    result = []
    
    def target():
        nonlocal result
        try:
            result = func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            return []
    
    # Create a separate thread for the function
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(target)
            future.result(timeout=METHOD_TIMEOUT)
            elapsed = time.time() - start_time
            print(f"Function {func.__name__} completed in {elapsed:.1f} seconds")
            return result
    except TimeoutError:
        print(f"⚠️ Function {func.__name__} timed out after {METHOD_TIMEOUT} seconds")
        return []

def detect_roundabouts_from_network_adaptive(streets_gdf, network_data, adaptive_params):
    """Adaptive network-based roundabout detection"""
    print("\n[METHOD 1] Adaptive network-based roundabout detection...")
    start_time = time.time()
    
    candidates = []
    intersections = network_data['intersections']
    
    try:
        from sklearn.cluster import DBSCAN
        
        if len(network_data['intersection_points']) > 10:
            print("Using adaptive DBSCAN for intersection clustering...")
            
            intersection_points = network_data['intersection_points']
            max_points = min(len(intersection_points), 30000)
            sample_points = intersection_points[:max_points]
            
            coords = np.array([[p.x, p.y] for p in sample_points])
            
            # Use adaptive parameters
            eps = adaptive_params['dbscan_eps']
            min_samples = adaptive_params['min_samples']
            
            print(f"Running DBSCAN with adaptive eps={eps}m, min_samples={min_samples}...")
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
            
            labels = clustering.labels_
            cluster_labels = set(labels)
            if -1 in cluster_labels:
                cluster_labels.remove(-1)
            
            print(f"Found {len(cluster_labels)} intersection clusters")
            
            # Process clusters with adaptive validation
            for label in cluster_labels:
                cluster_mask = labels == label
                cluster_points = [sample_points[i] for i, is_in_cluster in enumerate(cluster_mask) if is_in_cluster]
                
                if len(cluster_points) >= adaptive_params['min_samples']:
                    center, radius = minimum_enclosing_circle(cluster_points)
                    
                    if center and adaptive_params['min_radius'] <= radius <= adaptive_params['max_radius']:
                        circle = center.buffer(radius)
                        
                        # Adaptive validation
                        is_valid, validation_score, details = validate_roundabout_adaptive(
                            circle, center, radius, streets_gdf, cluster_points, adaptive_params)
                        
                        if is_valid:
                            # Count nearby streets
                            buffer = center.buffer(radius * 1.5)
                            nearby_streets = []
                            
                            try:
                                # Use spatial index if available
                                if hasattr(streets_gdf, 'sindex'):
                                    potential_indices = list(streets_gdf.sindex.intersection(buffer.bounds))
                                    for idx in potential_indices[:50]:  # Limit for performance
                                        if buffer.intersects(streets_gdf.iloc[idx].geometry):
                                            nearby_streets.append(streets_gdf.iloc[idx])
                                            if len(nearby_streets) >= 20:
                                                break
                                else:
                                    # Fallback without spatial index
                                    for idx, row in streets_gdf.head(1000).iterrows():
                                        if buffer.intersects(row.geometry):
                                            nearby_streets.append(row)
                                            if len(nearby_streets) >= 20:
                                                break
                            except Exception as e:
                                print(f"  Warning: Error counting nearby streets: {e}")
                                nearby_streets = []
                            
                            # Calculate adaptive confidence
                            intersection_factor = min(1.0, len(cluster_points) / 8)
                            street_factor = min(1.0, len(nearby_streets) / 5)
                            geometric_factor = validation_score
                            
                            # More forgiving confidence calculation
                            confidence = (0.4 * geometric_factor + 
                                         0.3 * street_factor + 
                                         0.3 * intersection_factor)
                            
                            # Use adaptive threshold
                            confidence_threshold = 0.4 if len(nearby_streets) >= 3 else 0.35
                            
                            if confidence >= confidence_threshold:
                                candidates.append({
                                    'geometry': circle,
                                    'radius': radius,
                                    'area': circle.area,
                                    'circularity': details['circularity'],
                                    'perimeter': circle.length,
                                    'intersection_density': float(len(cluster_points)),
                                    'connectivity': float(len(nearby_streets)),
                                    'nearby_streets': len(nearby_streets),
                                    'method': 'adaptive_network',
                                    'confidence': confidence,
                                    'quality_score': confidence,
                                    'validation_details': details
                                })
                                
                                print(f"  ✓ ADAPTIVE ROUNDABOUT: {radius:.0f}m radius, {len(nearby_streets)} streets, confidence {confidence:.2f}")
                            

    except ImportError:
        print("  sklearn not available for DBSCAN clustering")
    except Exception as e:
        print(f"  Error in adaptive network detection: {e}")
    
    print(f"Method 1 completed in {time.time() - start_time:.1f} seconds, found {len(candidates)} candidates")
    return candidates

def detect_curved_street_roundabouts_adaptive(streets_gdf, roundabout_name_indices, adaptive_params):
    """Adaptive curved street analysis"""
    print("\n[METHOD 2] Adaptive curved street analysis...")
    start_time = time.time()
    
    candidates = []
    curved_streets = []
    
    priority_indices = set(roundabout_name_indices if roundabout_name_indices else [])
    
    print(f"Analyzing curvature with adaptive thresholds...")
    
    # Sample a reasonable number of streets for analysis
    sample_size = min(50000, len(streets_gdf))
    streets_sample = streets_gdf.head(sample_size)
    
    for idx, row in streets_sample.iterrows():
        if idx % 10000 == 0:
            print(f"  Processed {idx}/{sample_size} streets")
            
        geom = row.geometry
        if geom.geom_type != 'LineString':
            continue
            
        coords = list(geom.coords)
        if len(coords) < 4:
            continue
            
        start_point = Point(coords[0])
        end_point = Point(coords[-1])
        straight_dist = start_point.distance(end_point)
        road_dist = geom.length
        
        if road_dist > 0:
            curvature = 1 - (straight_dist / road_dist)
            
            # Adaptive curvature thresholds
            is_priority = idx in priority_indices
            curvature_threshold = 0.3 if is_priority else 0.4
            length_min = adaptive_params['min_radius'] * 3
            length_max = adaptive_params['max_radius'] * 8
            
            if (curvature > curvature_threshold and 
                length_min < road_dist < length_max):
                
                curved_streets.append({
                    'geometry': geom,
                    'curvature': curvature,
                    'length': road_dist,
                    'idx': idx,
                    'is_priority': is_priority
                })
    
    print(f"Found {len(curved_streets)} curved street segments")
    
    if curved_streets:
        # Group curved streets
        spatial_index = rtree.index.Index()
        for idx, street in enumerate(curved_streets):
            spatial_index.insert(idx, street['geometry'].bounds)
        
        processed = set()
        curved_groups = []
        
        for idx, street in enumerate(curved_streets):
            if idx in processed or len(curved_groups) >= 50:
                continue
                
            street_geom = street['geometry']
            potential_matches = list(spatial_index.intersection(street_geom.bounds))[:30]
            
            group = [idx]
            processed.add(idx)
            is_priority = street['is_priority']
            
            for match_idx in potential_matches:
                if match_idx != idx and match_idx not in processed:
                    match_geom = curved_streets[match_idx]['geometry']
                    if street_geom.distance(match_geom) < adaptive_params['dbscan_eps']:
                        group.append(match_idx)
                        processed.add(match_idx)
                        if curved_streets[match_idx]['is_priority']:
                            is_priority = True
            
            if len(group) >= 2:  # At least 2 curved segments
                curved_groups.append({
                    'members': [curved_streets[i] for i in group],
                    'is_priority': is_priority
                })
        
        print(f"Found {len(curved_groups)} curved street groups")
        
        for group_data in curved_groups:
            group = group_data['members']
            geometries = [street['geometry'] for street in group]
            
            # Calculate center and radius
            all_points = []
            for geom in geometries:
                coords = list(geom.coords)
                for i in range(0, len(coords), max(1, len(coords)//5)):
                    all_points.append(Point(coords[i]))
            
            if len(all_points) >= 6:
                center, radius = minimum_enclosing_circle(all_points)
                
                if center and adaptive_params['min_radius'] <= radius <= adaptive_params['max_radius']:
                    roundabout = center.buffer(radius)
                    
                    # Adaptive validation
                    is_valid, validation_score, details = validate_roundabout_adaptive(
                        roundabout, center, radius, streets_gdf, all_points, adaptive_params)
                    
                    if is_valid:
                        avg_curvature = sum(street['curvature'] for street in group) / len(group)
                        
                        # Adaptive confidence calculation
                        curvature_factor = min(1.0, avg_curvature / 0.5)
                        group_size_factor = min(1.0, len(group) / 4)
                        priority_bonus = 0.2 if group_data['is_priority'] else 0
                        
                        confidence = (0.4 * validation_score + 
                                     0.3 * curvature_factor + 
                                     0.2 * group_size_factor + 
                                     priority_bonus)
                        
                        if confidence >= 0.5:
                            candidates.append({
                                'geometry': roundabout,
                                'radius': radius,
                                'area': roundabout.area,
                                'circularity': details['circularity'],
                                'perimeter': roundabout.length,
                                'curved_streets': len(group),
                                'avg_curvature': avg_curvature,
                                'method': 'adaptive_curved',
                                'confidence': confidence,
                                'quality_score': confidence,
                                'validation_details': details
                            })
                            
                            print(f"  ✓ ADAPTIVE CURVED ROUNDABOUT: {radius:.0f}m radius, {len(group)} segments, confidence {confidence:.2f}")
    
    print(f"Method 2 completed in {time.time() - start_time:.1f} seconds, found {len(candidates)} candidates")
    return candidates

def detect_junction_patterns_adaptive(streets_gdf, network_data, adaptive_params):
    """Adaptive junction pattern analysis"""
    print("\n[METHOD 3] Adaptive junction pattern analysis...")
    start_time = time.time()
    
    candidates = []
    intersection_points = network_data['intersection_points']
    
    if len(intersection_points) < 10:
        print("  Insufficient intersection data")
        return candidates
    
    try:
        from sklearn.cluster import DBSCAN
        
        print(f"Clustering with adaptive parameters...")
        
        max_points = min(len(intersection_points), 25000)
        sample_points = intersection_points[:max_points]
        coords = np.array([[p.x, p.y] for p in sample_points])
        
        # Use adaptive clustering parameters
        eps = adaptive_params['dbscan_eps'] * 0.8  # Slightly tighter for junction patterns
        min_samples = adaptive_params['min_samples'] + 1
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        
        labels = clustering.labels_
        cluster_labels = set(labels)
        if -1 in cluster_labels:
            cluster_labels.remove(-1)
            
        print(f"Found {len(cluster_labels)} intersection clusters")
        
        # Process all clusters (not just top ones)
        for label in cluster_labels:
            cluster_mask = labels == label
            cluster_points = [sample_points[i] for i, is_in_cluster in enumerate(cluster_mask) if is_in_cluster]
            
            if len(cluster_points) >= adaptive_params['min_samples']:
                center, radius = minimum_enclosing_circle(cluster_points)
                
                if center and adaptive_params['min_radius'] <= radius <= adaptive_params['max_radius']:
                    roundabout = center.buffer(radius)
                    
                    # Adaptive validation
                    is_valid, validation_score, details = validate_roundabout_adaptive(
                        roundabout, center, radius, streets_gdf, cluster_points, adaptive_params)
                    
                    if is_valid:
                        # Adaptive confidence calculation
                        intersection_factor = min(1.0, len(cluster_points) / 8)
                        geometric_factor = validation_score
                        
                        confidence = 0.6 * geometric_factor + 0.4 * intersection_factor
                        
                        if confidence >= 0.45:
                            candidates.append({
                                'geometry': roundabout,
                                'radius': radius,
                                'area': roundabout.area,
                                'circularity': details['circularity'],
                                'perimeter': roundabout.length,
                                'intersection_count': len(cluster_points),
                                'method': 'adaptive_junction',
                                'confidence': confidence,
                                'quality_score': confidence,
                                'validation_details': details
                            })
                            
                            print(f"  ✓ ADAPTIVE JUNCTION ROUNDABOUT: {radius:.0f}m radius, {len(cluster_points)} intersections, confidence {confidence:.2f}")
        
    except Exception as e:
        print(f"  Error in adaptive junction detection: {e}")
    
    print(f"Method 3 completed in {time.time() - start_time:.1f} seconds, found {len(candidates)} candidates")
    return candidates

def load_csv_roundabouts(csv_path='data/roundabout_export_10_1_2024.csv', streets_gdf=None):
    """Load roundabouts from CSV file and filter to study area"""
    print(f"\n🔍 Loading CSV roundabout data from {csv_path}...")
    
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return []
    
    try:
        # Load CSV data
        csv_roundabouts = pd.read_csv(csv_path)
        print(f"✓ Loaded {len(csv_roundabouts)} roundabouts from CSV")
        
        # Check columns
        print(f"CSV columns: {list(csv_roundabouts.columns)}")
        
        # Look for latitude/longitude columns (common variations)
        lat_cols = [col for col in csv_roundabouts.columns if 'lat' in col.lower()]
        lon_cols = [col for col in csv_roundabouts.columns if 'lon' in col.lower() or 'lng' in col.lower()]
        
        if not lat_cols or not lon_cols:
            print("❌ Could not find latitude/longitude columns in CSV")
            return []
        
        lat_col = lat_cols[0]
        lon_col = lon_cols[0]
        print(f"Using columns: {lat_col}, {lon_col}")
        
        # Remove rows with missing coordinates
        csv_roundabouts = csv_roundabouts.dropna(subset=[lat_col, lon_col])
        print(f"After removing missing coordinates: {len(csv_roundabouts)} roundabouts")
        
        # Convert to GeoDataFrame
        geometry = gpd.points_from_xy(csv_roundabouts[lon_col], csv_roundabouts[lat_col])
        csv_gdf = gpd.GeoDataFrame(csv_roundabouts, geometry=geometry, crs="EPSG:4326")
        
        # Filter to study area if we have street data
        if streets_gdf is not None:
            print("🔍 Filtering CSV roundabouts to study area...")
            
            # Get bounds of street data (in WGS84)
            streets_wgs84 = streets_gdf.to_crs("EPSG:4326")
            bounds = streets_wgs84.total_bounds
            
            # Create bounding box with some margin
            margin = 0.01  # degrees
            bbox = [
                bounds[0] - margin,  # min_lon
                bounds[1] - margin,  # min_lat
                bounds[2] + margin,  # max_lon
                bounds[3] + margin   # max_lat
            ]
            
            # Filter CSV roundabouts to bounding box
            mask = (
                (csv_gdf[lon_col] >= bbox[0]) & 
                (csv_gdf[lon_col] <= bbox[2]) &
                (csv_gdf[lat_col] >= bbox[1]) & 
                (csv_gdf[lat_col] <= bbox[3])
            )
            
            csv_filtered = csv_gdf[mask].copy()
            print(f"✓ Found {len(csv_filtered)} CSV roundabouts in study area")
            
            if len(csv_filtered) > 0:
                # Convert to same CRS as streets
                csv_filtered = csv_filtered.to_crs(streets_gdf.crs)
                return csv_filtered
            else:
                print("⚠️ No CSV roundabouts found in study area")
                return []
        else:
            print("✓ Returning all CSV roundabouts (no study area filtering)")
            return csv_gdf
            
    except Exception as e:
        print(f"❌ Error loading CSV roundabouts: {e}")
        return []

def extract_street_name_roundabouts(streets_gdf, roundabout_name_indices):
    """Extract actual roundabout geometries from streets with roundabout names"""
    print(f"\n🔍 Extracting roundabouts from {len(roundabout_name_indices)} street names...")
    
    street_name_roundabouts = []
    
    if not roundabout_name_indices:
        return street_name_roundabouts
    
    # Get streets with roundabout names
    roundabout_streets = streets_gdf.iloc[roundabout_name_indices].copy()
    
    # Group nearby roundabout streets
    spatial_index = rtree.index.Index()
    for idx, (orig_idx, row) in enumerate(roundabout_streets.iterrows()):
        spatial_index.insert(idx, row.geometry.bounds)
    
    processed = set()
    
    for idx, (orig_idx, row) in enumerate(roundabout_streets.iterrows()):
        if idx in processed:
            continue
            
        # Find nearby roundabout streets
        potential_matches = list(spatial_index.intersection(row.geometry.bounds))
        group_geoms = []
        group_names = []
        
        for match_idx in potential_matches:
            if match_idx not in processed:
                match_row = roundabout_streets.iloc[match_idx]
                if row.geometry.distance(match_row.geometry) < 100:  # Within 100m
                    group_geoms.append(match_row.geometry)
                    group_names.append(match_row.get('ST_NAME', 'Unknown'))
                    processed.add(match_idx)
        
        if len(group_geoms) >= 1:
            # Create a roundabout from the grouped streets
            try:
                # Get all points from the street segments
                all_points = []
                for geom in group_geoms:
                    coords = list(geom.coords)
                    for coord in coords:
                        all_points.append(Point(coord))
                
                if len(all_points) >= 3:
                    # Calculate center and radius
                    center, radius = minimum_enclosing_circle(all_points)
                    
                    if center and 15 <= radius <= 100:  # Reasonable roundabout size
                        roundabout_geom = center.buffer(radius)
                        
                        # Get unique street names
                        unique_names = list(set(group_names))
                        street_name = unique_names[0] if unique_names else 'Unknown'
                        
                        street_name_roundabouts.append({
                            'geometry': roundabout_geom,
                            'radius': radius,
                            'area': roundabout_geom.area,
                            'street_name': street_name,
                            'street_count': len(group_geoms),
                            'method': 'street_name',
                            'confidence': 0.9,  # High confidence for named roundabouts
                            'source': 'street_names'
                        })
                        
                        print(f"  ✓ STREET NAME ROUNDABOUT: '{street_name}' - {radius:.0f}m radius")
                        
            except Exception as e:
                print(f"  Warning: Error processing street group: {e}")
    
    print(f"✓ Extracted {len(street_name_roundabouts)} roundabouts from street names")
    return street_name_roundabouts

def create_html_map(detected_roundabouts, csv_roundabouts, street_name_roundabouts, streets_gdf):
    """Create an interactive HTML map showing all roundabouts"""
    print("\n🗺️ Creating interactive HTML map...")
    
    try:
        import folium
        from folium import plugins
    except ImportError:
        print("❌ folium not available. Install with: pip install folium")
        return
    
    # Calculate map center
    if len(detected_roundabouts) > 0:
        # Use detected roundabouts for center
        centroids = [r['geometry'].centroid for r in detected_roundabouts]
        center_lat = np.mean([c.y for c in centroids])
        center_lon = np.mean([c.x for c in centroids])
        
        # Convert to WGS84 for map
        from shapely.geometry import Point
        center_point = Point(center_lon, center_lat)
        center_gdf = gpd.GeoDataFrame([1], geometry=[center_point], crs=streets_gdf.crs)
        center_wgs84 = center_gdf.to_crs("EPSG:4326")
        map_center = [center_wgs84.geometry[0].y, center_wgs84.geometry[0].x]
    elif len(csv_roundabouts) > 0:
        # Use CSV roundabouts for center
        csv_wgs84 = csv_roundabouts.to_crs("EPSG:4326") if csv_roundabouts.crs != "EPSG:4326" else csv_roundabouts
        map_center = [csv_wgs84.geometry.centroid.y.mean(), csv_wgs84.geometry.centroid.x.mean()]
    else:
        # Use street data bounds
        streets_wgs84 = streets_gdf.to_crs("EPSG:4326")
        bounds = streets_wgs84.total_bounds
        map_center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
    
    # Create map
    m = folium.Map(location=map_center, zoom_start=12, tiles='OpenStreetMap')
    
    # Add different tile layers
    folium.TileLayer('cartodbpositron', name='CartoDB Positron').add_to(m)
    folium.TileLayer('cartodbdark_matter', name='CartoDB Dark').add_to(m)
    
    # Add detected roundabouts
    if detected_roundabouts:
        print(f"Adding {len(detected_roundabouts)} detected roundabouts to map...")
        detected_gdf = gpd.GeoDataFrame(detected_roundabouts, crs=streets_gdf.crs)
        detected_wgs84 = detected_gdf.to_crs("EPSG:4326")
        
        for idx, row in detected_wgs84.iterrows():
            center = row.geometry.centroid
            radius_m = row.get('radius', 30)
            method = row.get('method', 'unknown')
            confidence = row.get('confidence', 0)
            
            # Color by method
            method_colors = {
                'adaptive_network': '#FF0000',      # Red
                'adaptive_curved': '#00FF00',       # Green
                'adaptive_junction': '#800080'      # Purple
            }
            color = method_colors.get(method, '#FFA500')  # Orange default
            
            # Add circle
            folium.Circle(
                location=[center.y, center.x],
                radius=radius_m,
                popup=f"""
                <b>Detected Roundabout #{idx+1}</b><br>
                Method: {method}<br>
                Radius: {radius_m:.1f}m<br>
                Confidence: {confidence:.2f}<br>
                Circularity: {row.get('circularity', 0):.2f}
                """,
                color=color,
                weight=3,
                fillColor=color,
                fillOpacity=0.3
            ).add_to(m)
    
    # Add street name roundabouts
    if street_name_roundabouts:
        print(f"Adding {len(street_name_roundabouts)} street name roundabouts to map...")
        streetname_gdf = gpd.GeoDataFrame(street_name_roundabouts, crs=streets_gdf.crs)
        streetname_wgs84 = streetname_gdf.to_crs("EPSG:4326")
        
        for idx, row in streetname_wgs84.iterrows():
            center = row.geometry.centroid
            radius_m = row.get('radius', 30)
            street_name = row.get('street_name', 'Unknown')
            
            # Add circle for street name roundabouts
            folium.Circle(
                location=[center.y, center.x],
                radius=radius_m,
                popup=f"""
                <b>Street Name Roundabout</b><br>
                Street: {street_name}<br>
                Radius: {radius_m:.1f}m<br>
                Source: Street Names<br>
                Confidence: High
                """,
                color='#0000FF',  # Blue
                weight=4,
                fillColor='#0000FF',
                fillOpacity=0.4
            ).add_to(m)
    
    # Add CSV roundabouts
    if len(csv_roundabouts) > 0:
        print(f"Adding {len(csv_roundabouts)} CSV roundabouts to map...")
        csv_wgs84 = csv_roundabouts.to_crs("EPSG:4326") if csv_roundabouts.crs != "EPSG:4326" else csv_roundabouts
        
        for idx, row in csv_wgs84.iterrows():
            # Add marker for CSV roundabouts
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=8,
                popup=f"""
                <b>CSV Roundabout #{idx+1}</b><br>
                Source: Global Database<br>
                Lat: {row.geometry.y:.6f}<br>
                Lon: {row.geometry.x:.6f}
                """,
                color='#FFD700',  # Gold
                weight=2,
                fillColor='#FFD700',
                fillOpacity=0.8
            ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 120px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <p><b>Roundabout Legend</b></p>
    <p><i class="fa fa-circle" style="color:#FF0000"></i> Network Detection</p>
    <p><i class="fa fa-circle" style="color:#00FF00"></i> Curved Streets</p>
    <p><i class="fa fa-circle" style="color:#800080"></i> Junction Pattern</p>
    <p><i class="fa fa-circle" style="color:#0000FF"></i> Street Names</p>
    <p><i class="fa fa-circle" style="color:#FFD700"></i> CSV Database</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save map
    map_filename = 'roundabout_detection_map.html'
    m.save(map_filename)
    print(f"✅ Interactive map saved as: {map_filename}")
    
    return map_filename

def create_comprehensive_geojson(detected_roundabouts, csv_roundabouts, street_name_roundabouts, streets_gdf, output_filename='comprehensive_roundabouts_presentation.geojson'):
    """Create a comprehensive GeoJSON file with all roundabouts for final presentation"""
    print(f"\n💾 Creating comprehensive GeoJSON file: {output_filename}...")
    
    try:
        # Ensure all data is in WGS84 for GeoJSON
        all_features = []
        
        # Process CSV roundabouts (yellow points in HTML) - these get is_roundabout = 1
        if len(csv_roundabouts) > 0:
            print(f"Processing {len(csv_roundabouts)} CSV roundabouts (yellow points)...")
            csv_wgs84 = csv_roundabouts.to_crs("EPSG:4326") if csv_roundabouts.crs != "EPSG:4326" else csv_roundabouts.copy()
            
            for idx, row in csv_wgs84.iterrows():
                properties = {
                    'id': f'csv_roundabout_{idx+1}',
                    'source': 'global_database',
                    'type': 'roundabout',
                    'method': 'csv_database',
                    'color_in_map': 'yellow',
                    'is_roundabout': 1,  # Yellow points are confirmed roundabouts
                    'confidence': 1.0,   # High confidence from global database
                    'lat': row.geometry.y,
                    'lon': row.geometry.x,
                    'radius': 25.0,      # Default radius for CSV points
                    'description': 'Roundabout from global CSV database'
                }
                
                # Add all original CSV columns
                for col in csv_wgs84.columns:
                    if col not in ['geometry'] and pd.notna(row[col]):
                        value = row[col]
                        if isinstance(value, (int, float, str, bool)):
                            properties[f'csv_{col}'] = value
                        else:
                            properties[f'csv_{col}'] = str(value)
                
                feature = {
                    'type': 'Feature',
                    'properties': properties,
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [row.geometry.x, row.geometry.y]
                    }
                }
                all_features.append(feature)
        
        # Process detected roundabouts (red, green, purple in HTML) - these get is_roundabout = 1
        if detected_roundabouts:
            print(f"Processing {len(detected_roundabouts)} detected roundabouts...")
            detected_gdf = gpd.GeoDataFrame(detected_roundabouts, crs=streets_gdf.crs)
            detected_wgs84 = detected_gdf.to_crs("EPSG:4326")
            
            for idx, row in detected_wgs84.iterrows():
                method = row.get('method', 'unknown')
                
                # Map method to color
                method_colors = {
                    'adaptive_network': 'red',
                    'adaptive_curved': 'green', 
                    'adaptive_junction': 'purple'
                }
                color = method_colors.get(method, 'orange')
                
                center = row.geometry.centroid
                properties = {
                    'id': f'detected_roundabout_{idx+1}',
                    'source': 'algorithmic_detection',
                    'type': 'roundabout',
                    'method': method,
                    'color_in_map': color,
                    'is_roundabout': 1,  # Detected roundabouts are confirmed
                    'confidence': float(row.get('confidence', 0.5)),
                    'lat': center.y,
                    'lon': center.x,
                    'radius': float(row.get('radius', 30)),
                    'circularity': float(row.get('circularity', 0)),
                    'area': float(row.get('area', 0)),
                    'perimeter': float(row.get('perimeter', 0)),
                    'quality_score': float(row.get('quality_score', 0)),
                    'description': f'Roundabout detected using {method} method'
                }
                
                # Add method-specific properties
                if 'intersection_density' in row:
                    properties['intersection_density'] = float(row['intersection_density'])
                if 'connectivity' in row:
                    properties['connectivity'] = float(row['connectivity'])
                if 'curved_streets' in row:
                    properties['curved_streets'] = int(row['curved_streets'])
                if 'avg_curvature' in row:
                    properties['avg_curvature'] = float(row['avg_curvature'])
                if 'intersection_count' in row:
                    properties['intersection_count'] = int(row['intersection_count'])
                
                feature = {
                    'type': 'Feature',
                    'properties': properties,
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [center.x, center.y]
                    }
                }
                all_features.append(feature)
        
        # Process street name roundabouts (blue in HTML) - these get is_roundabout = 1
        if street_name_roundabouts:
            print(f"Processing {len(street_name_roundabouts)} street name roundabouts...")
            streetname_gdf = gpd.GeoDataFrame(street_name_roundabouts, crs=streets_gdf.crs)
            streetname_wgs84 = streetname_gdf.to_crs("EPSG:4326")
            
            for idx, row in streetname_wgs84.iterrows():
                center = row.geometry.centroid
                properties = {
                    'id': f'street_name_roundabout_{idx+1}',
                    'source': 'street_names',
                    'type': 'roundabout',
                    'method': 'street_name_analysis',
                    'color_in_map': 'blue',
                    'is_roundabout': 1,  # Street name roundabouts are confirmed
                    'confidence': float(row.get('confidence', 0.9)),
                    'lat': center.y,
                    'lon': center.x,
                    'radius': float(row.get('radius', 30)),
                    'area': float(row.get('area', 0)),
                    'street_name': row.get('street_name', 'Unknown'),
                    'street_count': int(row.get('street_count', 1)),
                    'description': f'Roundabout identified from street name: {row.get("street_name", "Unknown")}'
                }
                
                feature = {
                    'type': 'Feature',
                    'properties': properties,
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [center.x, center.y]
                    }
                }
                all_features.append(feature)
        
        # Create comprehensive GeoJSON structure
        geojson_data = {
            'type': 'FeatureCollection',
            'name': 'Comprehensive_Roundabouts_Final_Presentation',
            'crs': {
                'type': 'name',
                'properties': {
                    'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'
                }
            },
            'metadata': {
                'title': 'Comprehensive Roundabout Detection Results',
                'description': 'Final presentation dataset with all detected roundabouts',
                'creation_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'total_roundabouts': len(all_features),
                'csv_roundabouts': len(csv_roundabouts) if csv_roundabouts is not None else 0,
                'detected_roundabouts': len(detected_roundabouts) if detected_roundabouts else 0,
                'street_name_roundabouts': len(street_name_roundabouts) if street_name_roundabouts else 0,
                'legend': {
                    'yellow': 'CSV Database Roundabouts',
                    'red': 'Network-based Detection',
                    'green': 'Curved Street Detection', 
                    'purple': 'Junction Pattern Detection',
                    'blue': 'Street Name Roundabouts'
                }
            },
            'features': all_features
        }
        
        # Save to file
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, indent=2, ensure_ascii=False)
        
        # Print comprehensive summary
        print(f"✅ Successfully created comprehensive GeoJSON: {output_filename}")
        print(f"📍 File location: {os.path.abspath(output_filename)}")
        print(f"\n📊 FINAL PRESENTATION SUMMARY:")
        print(f"   🎯 Total roundabouts: {len(all_features)}")
        print(f"   🟡 CSV roundabouts (yellow): {len(csv_roundabouts) if csv_roundabouts is not None else 0}")
        print(f"   🔴 Network detection (red): {len([f for f in all_features if f['properties'].get('color_in_map') == 'red'])}")
        print(f"   🟢 Curved detection (green): {len([f for f in all_features if f['properties'].get('color_in_map') == 'green'])}")
        print(f"   🟣 Junction detection (purple): {len([f for f in all_features if f['properties'].get('color_in_map') == 'purple'])}")
        print(f"   🔵 Street names (blue): {len([f for f in all_features if f['properties'].get('color_in_map') == 'blue'])}")
        print(f"   ✅ All features have 'is_roundabout': 1")
        
        if len(all_features) > 0:
            # Calculate bounding box
            lats = [f['geometry']['coordinates'][1] for f in all_features]
            lons = [f['geometry']['coordinates'][0] for f in all_features]
            print(f"\n📍 Geographic Coverage:")
            print(f"   Latitude range: {min(lats):.6f}° to {max(lats):.6f}°")
            print(f"   Longitude range: {min(lons):.6f}° to {max(lons):.6f}°")
        
        print(f"\n🎁 PRESENTATION-READY FEATURES:")
        print(f"   • All roundabouts marked with 'is_roundabout': 1")
        print(f"   • Color coding matches HTML map")
        print(f"   • Complete metadata and descriptions")
        print(f"   • WGS84 coordinate system")
        print(f"   • Ready for GIS import and analysis")
        
        return output_filename
        
    except Exception as e:
        print(f"❌ Error creating comprehensive GeoJSON: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main detection pipeline with adaptive validation and HTML map generation"""
    overall_start = time.time()
    
    # Load and analyze data
    try:
        print("🔍 Starting street data loading...")
        streets_gdf = load_and_analyze_street_data()
        print("✅ Street data loaded successfully")
    except Exception as e:
        print(f"❌ DATA LOADING ERROR: {e}")
        return
    
    try:
        print("🔍 Creating spatial index...")
        if not hasattr(streets_gdf, 'sindex'):
            streets_gdf = streets_gdf.copy()
        print("✅ Spatial index ready")
    except Exception as e:
        print(f"❌ SPATIAL INDEX ERROR: {e}")
        return
    
    print("\n" + "="*70)
    print("RUNNING ADAPTIVE ROUNDABOUT DETECTION WITH CSV INTEGRATION")
    print("="*70)
    
    # Check for roundabout names
    roundabout_name_indices = []
    if USE_ST_NAME_HINTS:
        try:
            print("🔍 Checking for roundabout name hints...")
            roundabout_name_indices = check_for_roundabout_names(streets_gdf)
            print(f"✅ Found {len(roundabout_name_indices)} streets with roundabout-related names")
        except Exception as e:
            print(f"⚠️ WARNING: Error checking street names: {e}")
    
    # Extract roundabouts from street names
    street_name_roundabouts = []
    if roundabout_name_indices:
        try:
            street_name_roundabouts = extract_street_name_roundabouts(streets_gdf, roundabout_name_indices)
        except Exception as e:
            print(f"⚠️ WARNING: Error extracting street name roundabouts: {e}")
    
    # Load CSV roundabouts
    csv_roundabouts = []
    try:
        print("🔍 Loading CSV roundabout database...")
        csv_roundabouts = load_csv_roundabouts('data/roundabout_export_10_1_2024.csv', streets_gdf)
    except Exception as e:
        print(f"⚠️ WARNING: Error loading CSV roundabouts: {e}")
    
    # Build network graph
    try:
        print("🔍 Building street network graph...")
        network_data = build_street_network_graph(streets_gdf)
        print(f"✅ Network graph built with {len(network_data['intersections'])} intersections")
    except Exception as e:
        print(f"❌ NETWORK GRAPH ERROR: {e}")
        network_data = {
            'intersections': [],
            'intersection_points': [],
            'connections': defaultdict(set),
            'high_connectivity_streets': {}
        }
    
    # Analyze data characteristics for adaptive parameters
    adaptive_params = analyze_data_characteristics(streets_gdf, network_data['intersections'])
    
    all_candidates = []
    
    # Method 1: Adaptive network-based detection
    try:
        print("🔍 Starting Method 1: Adaptive network-based detection...")
        network_candidates = timeout_handler(detect_roundabouts_from_network_adaptive, streets_gdf, network_data, adaptive_params)
        all_candidates.extend(network_candidates)
        print(f"✅ Method 1 found {len(network_candidates)} adaptive network candidates")
    except Exception as e:
        print(f"❌ Method 1 failed: {e}")
    
    # Method 2: Adaptive curved street analysis
    try:
        print("🔍 Starting Method 2: Adaptive curved street analysis...")
        curved_candidates = timeout_handler(detect_curved_street_roundabouts_adaptive, streets_gdf, roundabout_name_indices, adaptive_params)
        all_candidates.extend(curved_candidates)
        print(f"✅ Method 2 found {len(curved_candidates)} adaptive curved candidates")
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")
    
    # Method 3: Adaptive junction pattern analysis
    try:
        print("🔍 Starting Method 3: Adaptive junction pattern analysis...")
        junction_candidates = timeout_handler(detect_junction_patterns_adaptive, streets_gdf, network_data, adaptive_params)
        all_candidates.extend(junction_candidates)
        print(f"✅ Method 3 found {len(junction_candidates)} adaptive junction candidates")
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")
    
    print(f"\n🎯 TOTAL DETECTION RESULTS:")
    print(f"   Adaptive Detection: {len(all_candidates)} candidates")
    print(f"   Street Names: {len(street_name_roundabouts)} roundabouts")
    print(f"   CSV Database: {len(csv_roundabouts)} roundabouts")
    
    # Process detected roundabouts
    final_candidates = []
    if all_candidates:
        # Sort by quality score
        ranked_candidates = sorted(all_candidates, key=lambda x: x.get('quality_score', 0), reverse=True)
        
        # Apply reasonable confidence threshold
        filtered_candidates = [c for c in ranked_candidates if c.get('confidence', 0) >= 0.4]
        
        # Remove duplicates
        for candidate in filtered_candidates:
            is_duplicate = any(
                candidate['geometry'].centroid.distance(existing['geometry'].centroid) < 60
                for existing in final_candidates
            )
            if not is_duplicate:
                final_candidates.append(candidate)
                
            if len(final_candidates) >= MAX_RESULTS:
                break
    
    # Create HTML map with all roundabout sources
    try:
        map_filename = create_html_map(final_candidates, csv_roundabouts, street_name_roundabouts, streets_gdf)
        print(f"\n🗺️ Interactive map created: {map_filename}")
    except Exception as e:
        print(f"⚠️ Could not create HTML map: {e}")
    
    # Create comprehensive GeoJSON for final presentation
    try:
        comprehensive_geojson = create_comprehensive_geojson(
            final_candidates, csv_roundabouts, street_name_roundabouts, streets_gdf,
            'comprehensive_roundabouts_final_presentation.geojson'
        )
        if comprehensive_geojson:
            print(f"\n🎯 FINAL PRESENTATION FILE CREATED: {comprehensive_geojson}")
    except Exception as e:
        print(f"⚠️ Could not create comprehensive GeoJSON: {e}")
    
    # Print final summary
    print(f"\n📋 FINAL ROUNDABOUT SUMMARY:")
    print(f"   🔍 Detected (Adaptive): {len(final_candidates)}")
    print(f"   📛 Street Names: {len(street_name_roundabouts)}")
    print(f"   📊 CSV Database: {len(csv_roundabouts)}")
    print(f"   🎯 Total Roundabouts: {len(final_candidates) + len(street_name_roundabouts) + len(csv_roundabouts)}")
    
    # Save results
    if final_candidates or street_name_roundabouts or csv_roundabouts:
        try:
            # Combine all roundabouts
            all_roundabouts = []
            
            # Add detected roundabouts
            for i, r in enumerate(final_candidates):
                r_copy = r.copy()
                r_copy['source'] = 'detection'
                r_copy['id'] = f"detected_{i+1}"
                all_roundabouts.append(r_copy)
            
            # Add street name roundabouts
            for i, r in enumerate(street_name_roundabouts):
                r_copy = r.copy()
                r_copy['source'] = 'street_names'
                r_copy['id'] = f"street_name_{i+1}"
                all_roundabouts.append(r_copy)
            
            if all_roundabouts:
                # Save combined results
                roundabout_gdf = gpd.GeoDataFrame(all_roundabouts, crs=streets_gdf.crs)
                roundabout_gdf_wgs84 = roundabout_gdf.to_crs("EPSG:4326")
                
                geojson_data = {'type': 'FeatureCollection', 'features': []}
                
                for idx, row in roundabout_gdf_wgs84.iterrows():
                    properties = {
                        'id': row.get('id', f'roundabout_{idx+1}'),
                        'source': row.get('source', 'unknown'),
                        'method': row.get('method', 'unknown'),
                        'radius': float(row.get('radius', 0)) if pd.notna(row.get('radius', 0)) else 0.0,
                        'confidence': float(row.get('confidence', 0)) if pd.notna(row.get('confidence', 0)) else 0.0,
                        'street_name': row.get('street_name', ''),
                        'is_roundabout': 1
                    }
                    
                    feature = {
                        'type': 'Feature',
                        'geometry': row.geometry.__geo_interface__,
                        'properties': properties
                    }
                    geojson_data['features'].append(feature)
                
                with open('all_roundabouts.geojson', 'w') as f:
                    json.dump(geojson_data, f, indent=2)
                
                print("💾 All results saved: all_roundabouts.geojson")
                print(f"🎉 SUCCESS! Found {len(all_roundabouts)} total roundabouts!")
            
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")
    
    # Print file summary
    print(f"\n📁 OUTPUT FILES CREATED:")
    print(f"   🎯 comprehensive_roundabouts_final_presentation.geojson - FINAL PRESENTATION FILE")
    if len(csv_roundabouts) > 0:
        print(f"   🟡 Contains {len(csv_roundabouts)} CSV roundabouts (yellow points)")
    if final_candidates:
        print(f"   🎨 Contains {len(final_candidates)} detected roundabouts (red/green/purple)")
    if street_name_roundabouts:
        print(f"   🔵 Contains {len(street_name_roundabouts)} street name roundabouts (blue)")
    print(f"   🗺️ roundabout_detection_map.html - Interactive map")
    print(f"   📄 all_roundabouts.geojson - Combined results")
    
    total_time = time.time() - overall_start
    print(f"\n⏱️ Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print("🏁 Analysis complete!")

if __name__ == "__main__":
    try:
        main()
        print("\n⚡ ANALYSIS COMPLETE!")
        print("="*70)
    except KeyboardInterrupt:
        print("\n\n⚠️ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ UNEXPECTED ERROR: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)
