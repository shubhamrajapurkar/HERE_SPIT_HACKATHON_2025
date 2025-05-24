from flask import Blueprint, jsonify, request, send_file
import requests
import json
import re
import os
from osm2geojson import json2geojson
import tempfile
from datetime import datetime


overpass = Blueprint('overpass', __name__)

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
            response = requests.post(
                endpoint,
                data={'data': overpass_query},
                headers=headers,
                timeout=150
            )
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    return data
                except json.JSONDecodeError:
                    continue
            else:
                continue
                
        except requests.exceptions.RequestException:
            continue
    
    raise Exception("All Overpass API endpoints failed")

def process_geojson(data: dict, data_type: str):
    """Process OSM data and convert to GeoJSON"""
    try:
        elements = data.get('elements', [])
        if not elements:
            return None
        
        # Convert to GeoJSON
        geojson_data = json2geojson(data)
        features = geojson_data.get('features', [])
        
        if not features:
            return None
            
        return geojson_data
        
    except Exception as e:
        raise Exception(f"Error processing {data_type} GeoJSON: {e}")

# API Routes
@overpass.route('/areas')
def list_areas():
    """Get list of available Mumbai areas"""
    return jsonify({
        "available_areas": list(MUMBAI_AREAS.keys()),
        "default_bounds": MUMBAI_AREAS["mumbai_full"],
        "data_types": ["roads", "buildings", "hotels", "colleges", "all"]
    })

@overpass.route('/<data_type>/<area>')
def get_osm_data(data_type, area):
    """Extract specific data type from Mumbai area"""
    try:
        if area not in MUMBAI_AREAS:
            return jsonify({"error": f"Area '{area}' not found. Available areas: {list(MUMBAI_AREAS.keys())}"}), 400
        
        if data_type not in ["roads", "buildings", "hotels", "colleges", "all"]:
            return jsonify({"error": f"Data type '{data_type}' not supported. Available types: roads, buildings, hotels, colleges, all"}), 400
        
        # Generate Overpass query
        query = create_overpass_query(area, data_type)
        
        # Query Overpass API
        osm_data = query_overpass(query, data_type)
        
        # Process to GeoJSON
        geojson_data = process_geojson(osm_data, data_type)
        
        if not geojson_data:
            return jsonify({"message": f"No {data_type} found in {area}", "features": []}), 200
        
        # Add metadata
        response_data = {
            "type": "FeatureCollection",
            "metadata": {
                "area": area,
                "data_type": data_type,
                "feature_count": len(geojson_data.get('features', [])),
                "timestamp": datetime.now().isoformat(),
                "bounds": MUMBAI_AREAS[area]
            },
            "features": geojson_data.get('features', [])
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@overpass.route('/extract', methods=['POST'])
def extract_multiple_data():
    """Extract multiple data types from an area"""
    try:
        data = request.get_json()
        area = data.get('area', 'mumbai_full')
        data_types = data.get('data_types', ['roads', 'buildings'])
        
        if area not in MUMBAI_AREAS:
            return jsonify({"error": f"Area '{area}' not found"}), 400
        
        results = {}
        
        for data_type in data_types:
            try:
                query = create_overpass_query(area, data_type)
                osm_data = query_overpass(query, data_type)
                geojson_data = process_geojson(osm_data, data_type)
                
                if geojson_data:
                    results[data_type] = {
                        "feature_count": len(geojson_data.get('features', [])),
                        "features": geojson_data.get('features', [])
                    }
                else:
                    results[data_type] = {
                        "feature_count": 0,
                        "features": []
                    }
                    
            except Exception as e:
                results[data_type] = {
                    "error": str(e),
                    "feature_count": 0,
                    "features": []
                }
        
        response_data = {
            "area": area,
            "bounds": MUMBAI_AREAS[area],
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "total_features": sum(r.get('feature_count', 0) for r in results.values())
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@overpass.route('/compare/<data_type>')
def compare_areas(data_type):
    """Compare data type across different Mumbai areas"""
    try:
        areas = ["south_mumbai", "central_mumbai", "western_suburbs", "andheri", "bandra"]
        
        if data_type not in ["roads", "buildings", "hotels", "colleges"]:
            return jsonify({"error": f"Data type '{data_type}' not supported"}), 400
        
        area_results = {}
        
        for area in areas:
            try:
                query = create_overpass_query(area, data_type)
                osm_data = query_overpass(query, data_type)
                geojson_data = process_geojson(osm_data, data_type)
                
                area_results[area] = {
                    "feature_count": len(geojson_data.get('features', [])) if geojson_data else 0,
                    "bounds": MUMBAI_AREAS[area],
                    "features": geojson_data.get('features', []) if geojson_data else []
                }
                
            except Exception as e:
                area_results[area] = {
                    "error": str(e),
                    "feature_count": 0,
                    "bounds": MUMBAI_AREAS[area],
                    "features": []
                }
        
        # Sort by feature count
        sorted_areas = sorted(area_results.items(), key=lambda x: x[1].get('feature_count', 0), reverse=True)
        
        response_data = {
            "data_type": data_type,
            "timestamp": datetime.now().isoformat(),
            "area_comparison": dict(sorted_areas),
            "summary": {
                "total_areas": len(areas),
                "total_features": sum(r.get('feature_count', 0) for r in area_results.values()),
                "top_area": sorted_areas[0][0] if sorted_areas else None
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@overpass.route('/download/<data_type>/<area>')
def download_geojson(data_type, area):
    """Download GeoJSON file for specific data type and area"""
    try:
        if area not in MUMBAI_AREAS:
            return jsonify({"error": f"Area '{area}' not found"}), 400
        
        if data_type not in ["roads", "buildings", "hotels", "colleges", "all"]:
            return jsonify({"error": f"Data type '{data_type}' not supported"}), 400
        
        # Generate and query data
        query = create_overpass_query(area, data_type)
        osm_data = query_overpass(query, data_type)
        geojson_data = process_geojson(osm_data, data_type)
        
        if not geojson_data:
            return jsonify({"error": f"No {data_type} found in {area}"}), 404
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False, encoding='utf-8') as f:
            json.dump(geojson_data, f, indent=2, ensure_ascii=False)
            temp_path = f.name
        
        filename = f"mumbai_{area}_{data_type}.geojson"
        
        return send_file(
            temp_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/geo+json'
        )
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@overpass.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "available_areas": len(MUMBAI_AREAS),
        "supported_data_types": ["roads", "buildings", "hotels", "colleges", "all"]
    })
