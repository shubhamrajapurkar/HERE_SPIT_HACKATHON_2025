import requests
import json
import re
import os
from osm2geojson import json2geojson
from typing import Dict, List, Any
import glob
from pathlib import Path

# Mumbai bounding box coordinates (South, West, North, East)
MUMBAI_BOUNDS = [18.8800, 72.7800, 19.2600, 73.0300]

# Specific Mumbai areas with their bounding boxes
MUMBAI_AREAS = {
    "mumbai_full": [18.8800, 72.7800, 19.2600, 73.0300],
    "south_mumbai": [18.8800, 72.8000, 18.9800, 72.8600],
    "central_mumbai": [19.0000, 72.8200, 19.1000, 72.8800],
    "western_suburbs": [19.0500, 72.8000, 19.2600, 72.9000],
    "andheri": [19.1000, 72.8000, 19.1500, 72.8700],
    "bandra": [19.0400, 72.8100, 19.0700, 72.8400],
    "juhu": [19.0900, 72.8200, 19.1200, 72.8400],
    "powai": [19.1000, 72.8800, 19.1300, 72.9200],
    "goregaon": [19.1500, 72.8200, 19.1800, 72.8600],
    "malad": [19.1800, 72.8300, 19.2100, 72.8600]
}

def create_overpass_query(area_name: str, data_type: str) -> str:
    """Create specific Overpass QL queries for Mumbai data"""
    
    # Get bounding box for the area
    bounds = MUMBAI_AREAS.get(area_name.lower().replace(" ", "_"), MUMBAI_BOUNDS)
    bbox = f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}"
    
    # Define queries for different data types
    queries = {
        "roads": f"""
[out:json][timeout:90];
(
  way["highway"]({bbox});
  way["highway"~"^(primary|secondary|tertiary|residential|trunk|motorway|unclassified|service)$"]({bbox});
);
out geom;
""",
        
        "buildings": f"""
[out:json][timeout:90];
(
  way["building"]({bbox});
  relation["building"]({bbox});
  way["building"~"^(yes|residential|commercial|industrial|office|retail|house|apartments)$"]({bbox});
);
out geom;
""",
        
        "hotels": f"""
[out:json][timeout:90];
(
  node["tourism"="hotel"]({bbox});
  way["tourism"="hotel"]({bbox});
  node["tourism"="guest_house"]({bbox});
  way["tourism"="guest_house"]({bbox});
  node["amenity"="hotel"]({bbox});
  way["amenity"="hotel"]({bbox});
  node["name"~"hotel|lodge|inn|resort",i]({bbox});
  way["name"~"hotel|lodge|inn|resort",i]({bbox});
);
out geom;
""",
        
        "colleges": f"""
[out:json][timeout:90];
(
  node["amenity"="college"]({bbox});
  way["amenity"="college"]({bbox});
  node["amenity"="university"]({bbox});
  way["amenity"="university"]({bbox});
  node["amenity"="school"]({bbox});
  way["amenity"="school"]({bbox});
  node["name"~"college|university|institute|school",i]({bbox});
  way["name"~"college|university|institute|school",i]({bbox});
);
out geom;
""",
        
        "all": f"""
[out:json][timeout:120];
(
  // Roads
  way["highway"]({bbox});
  
  // Buildings  
  way["building"]({bbox});
  relation["building"]({bbox});
  
  // Hotels
  node["tourism"~"^(hotel|guest_house)$"]({bbox});
  way["tourism"~"^(hotel|guest_house)$"]({bbox});
  node["amenity"="hotel"]({bbox});
  way["amenity"="hotel"]({bbox});
  
  // Colleges
  node["amenity"~"^(college|university|school)$"]({bbox});
  way["amenity"~"^(college|university|school)$"]({bbox});
);
out geom;
"""
    }
    
    return queries.get(data_type.lower(), queries["all"])

def query_overpass(overpass_query: str, data_type: str = "data") -> dict:
    """Query Overpass API with multiple endpoints for reliability"""
    
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://lz4.overpass-api.de/api/interpreter", 
        "https://z.overpass-api.de/api/interpreter"
    ]
    
    headers = {
        "Accept": "application/json",
        "User-Agent": "Mumbai OSM Data Extractor/1.0"
    }
    
    for i, endpoint in enumerate(endpoints):
        try:
            print(f"🌐 Trying endpoint {i+1}/3: {endpoint.split('//')[1].split('/')[0]}")
            
            response = requests.post(
                endpoint,
                data={'data': overpass_query},
                headers=headers,
                timeout=150
            )
            
            print(f"📡 Response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    element_count = len(data.get('elements', []))
                    print(f"✅ Success! Found {element_count} {data_type} elements")
                    return data
                except json.JSONDecodeError as e:
                    print(f"❌ JSON decode error: {e}")
                    continue
            else:
                print(f"❌ HTTP {response.status_code}: {response.text[:200]}")
                continue
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Request error: {e}")
            continue
    
    raise Exception("❌ All Overpass API endpoints failed")

def save_geojson(data: dict, filename: str, data_type: str):
    """Save OSM data as GeoJSON with detailed logging"""
    
    try:
        elements = data.get('elements', [])
        if not elements:
            print(f"⚠️ No {data_type} elements found")
            return None, 0
        
        print(f"📊 Processing {len(elements)} {data_type} elements...")
        
        # Convert to GeoJSON
        geojson_data = json2geojson(data)
        features = geojson_data.get('features', [])
        
        if not features:
            print(f"⚠️ No GeoJSON features generated for {data_type}")
            return None, 0
        
        # Add data type and area information to each feature
        area_from_filename = filename.split('_')[1] if '_' in filename else 'unknown'
        for feature in features:
            feature['properties']['data_type'] = data_type
            feature['properties']['area'] = area_from_filename
            feature['properties']['extraction_timestamp'] = json.dumps({"extracted": True})
        
        # Save to file
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(geojson_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ {data_type.title()} GeoJSON saved to {filename}")
        print(f"📊 Features saved: {len(features)}")
        
        # Show sample data
        if features:
            sample = features[0]['properties']
            sample_info = []
            
            if 'name' in sample:
                sample_info.append(f"Name: {sample['name']}")
            if 'highway' in sample:
                sample_info.append(f"Highway: {sample['highway']}")
            if 'building' in sample:
                sample_info.append(f"Building: {sample['building']}")
            if 'amenity' in sample:
                sample_info.append(f"Amenity: {sample['amenity']}")
            if 'tourism' in sample:
                sample_info.append(f"Tourism: {sample['tourism']}")
                
            if sample_info:
                print(f"📍 Sample {data_type}: {' | '.join(sample_info)}")
        
        return geojson_data, len(features)
        
    except Exception as e:
        print(f"❌ Error saving {data_type} GeoJSON: {e}")
        
        # Save raw data as backup
        backup_filename = f"raw_{filename.replace('.geojson', '.json')}"
        try:
            with open(backup_filename, "w", encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"💾 Raw {data_type} data saved to {backup_filename}")
        except Exception as backup_error:
            print(f"❌ Failed to save backup: {backup_error}")
        
        return None, 0

def merge_geojson_files(pattern: str = "mumbai_*.geojson", output_filename: str = "mumbai_complete_merged.geojson") -> Dict[str, Any]:
    """
    Merge multiple GeoJSON files into a single comprehensive file.
    
    Args:
        pattern: File pattern to match GeoJSON files (default: "mumbai_*.geojson")
        output_filename: Name of the merged output file
    
    Returns:
        Dictionary with merge statistics
    """
    
    print(f"\n🔄 MERGING GEOJSON FILES")
    print("=" * 60)
    
    # Find all matching GeoJSON files
    geojson_files = glob.glob(pattern)
    
    if not geojson_files:
        print(f"❌ No GeoJSON files found matching pattern: {pattern}")
        return {"status": "failed", "reason": "no_files_found"}
    
    print(f"📁 Found {len(geojson_files)} GeoJSON files to merge:")
    for file in geojson_files:
        print(f"   📄 {file}")
    
    # Initialize merged GeoJSON structure
    merged_geojson = {
        "type": "FeatureCollection",
        "features": [],
        "properties": {
            "merged_from": geojson_files,
            "merge_timestamp": json.dumps({"merged": True}),
            "total_source_files": len(geojson_files)
        }
    }
    
    merge_stats = {
        "total_files": len(geojson_files),
        "successful_merges": 0,
        "failed_files": [],
        "features_by_type": {},
        "features_by_area": {},
        "total_features": 0
    }
    
    # Process each GeoJSON file
    for file_path in geojson_files:
        try:
            print(f"\n📖 Processing: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            
            features = geojson_data.get('features', [])
            if not features:
                print(f"⚠️ No features found in {file_path}")
                continue
            
            # Extract metadata from filename
            filename_parts = Path(file_path).stem.split('_')
            area = filename_parts[1] if len(filename_parts) > 1 else 'unknown'
            data_type = filename_parts[2] if len(filename_parts) > 2 else 'unknown'
            
            print(f"   📊 Features: {len(features)} | Area: {area} | Type: {data_type}")
            
            # Add source file information to each feature
            for feature in features:
                if 'properties' not in feature:
                    feature['properties'] = {}
                
                feature['properties']['source_file'] = file_path
                feature['properties']['source_area'] = area
                feature['properties']['source_data_type'] = data_type
                
                # Ensure data_type and area are set (in case they weren't added during extraction)
                if 'data_type' not in feature['properties']:
                    feature['properties']['data_type'] = data_type
                if 'area' not in feature['properties']:
                    feature['properties']['area'] = area
            
            # Add features to merged collection
            merged_geojson['features'].extend(features)
            
            # Update statistics
            merge_stats['successful_merges'] += 1
            merge_stats['total_features'] += len(features)
            
            # Count by type and area
            if data_type not in merge_stats['features_by_type']:
                merge_stats['features_by_type'][data_type] = 0
            merge_stats['features_by_type'][data_type] += len(features)
            
            if area not in merge_stats['features_by_area']:
                merge_stats['features_by_area'][area] = 0
            merge_stats['features_by_area'][area] += len(features)
            
            print(f"   ✅ Successfully processed {len(features)} features")
            
        except Exception as e:
            print(f"   ❌ Error processing {file_path}: {e}")
            merge_stats['failed_files'].append({"file": file_path, "error": str(e)})
    
    # Save merged GeoJSON
    if merge_stats['total_features'] > 0:
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(merged_geojson, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Merged GeoJSON saved to: {output_filename}")
            merge_stats['status'] = 'success'
            merge_stats['output_file'] = output_filename
            
        except Exception as e:
            print(f"\n❌ Error saving merged file: {e}")
            merge_stats['status'] = 'failed'
            merge_stats['error'] = str(e)
    else:
        print(f"\n⚠️ No features to merge!")
        merge_stats['status'] = 'failed'
        merge_stats['error'] = 'no_features'
    
    # Print merge summary
    print(f"\n" + "=" * 60)
    print("📊 MERGE SUMMARY")
    print("=" * 60)
    print(f"📁 Total files processed: {merge_stats['total_files']}")
    print(f"✅ Successful merges: {merge_stats['successful_merges']}")
    print(f"❌ Failed files: {len(merge_stats['failed_files'])}")
    print(f"🎯 Total features merged: {merge_stats['total_features']}")
    
    if merge_stats['features_by_type']:
        print(f"\n📋 Features by Data Type:")
        for data_type, count in sorted(merge_stats['features_by_type'].items()):
            print(f"   🏷️ {data_type.title()}: {count}")
    
    if merge_stats['features_by_area']:
        print(f"\n🗺️ Features by Area:")
        for area, count in sorted(merge_stats['features_by_area'].items(), key=lambda x: x[1], reverse=True):
            area_name = area.replace('_', ' ').title()
            print(f"   📍 {area_name}: {count}")
    
    if merge_stats['failed_files']:
        print(f"\n❌ Failed Files:")
        for failed in merge_stats['failed_files']:
            print(f"   📄 {failed['file']}: {failed['error']}")
    
    return merge_stats

def extract_mumbai_data(area: str = "mumbai_full", data_types: list = ["roads", "buildings", "hotels", "colleges"]):
    """Extract specific data types from Mumbai area and return GeoJSON data for merging"""
    
    print(f"🏙️ Extracting data from {area.replace('_', ' ').title()}")
    print(f"📋 Data types: {', '.join(data_types)}")
    print("=" * 60)
    
    results = {}
    geojson_data_collection = []
    
    for data_type in data_types:
        try:
            print(f"\n🔍 Extracting {data_type}...")
            
            # Generate Overpass query
            query = create_overpass_query(area, data_type)
            print(f"📝 Query generated for {data_type}")
            
            # Query Overpass API
            print(f"📡 Querying Overpass API for {data_type}...")
            osm_data = query_overpass(query, data_type)
            
            # Save as GeoJSON and collect data
            filename = f"mumbai_{area}_{data_type}.geojson"
            geojson_data, feature_count = save_geojson(osm_data, filename, data_type)
            
            if geojson_data:
                geojson_data_collection.append({
                    'data': geojson_data,
                    'area': area,
                    'type': data_type,
                    'filename': filename
                })
            
            results[data_type] = feature_count or 0
            
        except Exception as e:
            print(f"❌ Failed to extract {data_type}: {e}")
            results[data_type] = 0
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 60)
    
    for data_type, count in results.items():
        status = "✅" if count > 0 else "❌"
        print(f"{status} {data_type.title()}: {count} features")
    
    total_features = sum(results.values())
    print(f"\n🎯 Total features extracted: {total_features}")
    
    return results, geojson_data_collection

def extract_mumbai_area_comparison():
    """Extract colleges from different Mumbai areas for comparison"""
    
    areas = ["south_mumbai", "central_mumbai", "western_suburbs", "andheri", "bandra"]
    
    print("🏙️ Mumbai Area Comparison - Colleges Distribution")
    print("=" * 60)
    
    area_results = {}
    
    for area in areas:
        try:
            print(f"\n📍 Extracting colleges from {area.replace('_', ' ').title()}...")
            
            query = create_overpass_query(area, "colleges")
            osm_data = query_overpass(query, "colleges")
            
            filename = f"mumbai_{area}_colleges.geojson"
            geojson_data, feature_count = save_geojson(osm_data, filename, "colleges")
            area_results[area] = feature_count or 0
            
        except Exception as e:
            print(f"❌ Failed for {area}: {e}")
            area_results[area] = 0
    
    # Area comparison summary
    print("\n" + "=" * 60)
    print("📊 MUMBAI AREAS - COLLEGES COMPARISON")
    print("=" * 60)
    
    for area, count in sorted(area_results.items(), key=lambda x: x[1], reverse=True):
        area_name = area.replace('_', ' ').title()
        print(f"🏫 {area_name:<20}: {count:>3} colleges")
    
    return area_results

# Main execution
if __name__ == "__main__":
    
    print("🇮🇳 MUMBAI OSM DATA EXTRACTOR WITH MERGER")
    print("=" * 60)
    
    # Option 1: Extract all data types from full Mumbai
    print("\n1️⃣ EXTRACTING ALL DATA FROM MUMBAI")
    try:
        extract_mumbai_data(
            area="mumbai_full",
            data_types=["roads", "buildings", "hotels", "colleges"]
        )
    except Exception as e:
        print(f"❌ Full Mumbai extraction failed: {e}")
    
    # Option 2: Extract from specific area (Andheri)
    print("\n\n2️⃣ EXTRACTING DATA FROM ANDHERI")
    try:
        extract_mumbai_data(
            area="andheri", 
            data_types=["hotels", "colleges"]
        )
    except Exception as e:
        print(f"❌ Andheri extraction failed: {e}")
    
    # Option 3: Compare colleges across Mumbai areas
    print("\n\n3️⃣ MUMBAI AREAS COMPARISON")
    try:
        extract_mumbai_area_comparison()
    except Exception as e:
        print(f"❌ Area comparison failed: {e}")
    
    # Option 4: Merge all GeoJSON files
    print("\n\n4️⃣ MERGING ALL GEOJSON FILES")
    try:
        merge_stats = merge_geojson_files(
            pattern="mumbai_*.geojson",
            output_filename="mumbai_complete_merged.geojson"
        )
        
        if merge_stats.get('status') == 'success':
            print(f"\n🎉 All data successfully merged into: {merge_stats['output_file']}")
            print(f"📊 Total features in merged file: {merge_stats['total_features']}")
        else:
            print(f"\n❌ Merge failed: {merge_stats.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Merge process failed: {e}")
    
    print("\n🎉 Processing completed!")
    print("💡 Individual GeoJSON files are available for specific use cases.")
    print("🔗 The merged file 'mumbai_complete_merged.geojson' contains all data.")
    print("🗺️ You can open these files in QGIS, ArcGIS, or any GIS software.")