import json
import google.generativeai as genai
from typing import Dict, List, Any, Tuple
import hashlib
from collections import defaultdict

class GeoJSONComparator:
    def __init__(self, api_key: str):  # Fixed constructor name
        """Initialize the GeoJSON comparator with Gemini API key."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def load_geojson(self, file_path: str) -> Dict[str, Any]:
        """Load GeoJSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not data:
                    raise ValueError(f"File {file_path} is empty or invalid")
                if 'features' not in data:
                    raise ValueError(f"File {file_path} is not a valid GeoJSON FeatureCollection")
                print(f"Loaded {file_path}: {len(data.get('features', []))} features")
                return data
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {file_path}: {e}")
        except Exception as e:
            raise Exception(f"Error loading {file_path}: {e}")
    
    def extract_feature_summary(self, geojson: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key information from GeoJSON for comparison."""
        if not geojson:
            raise ValueError("GeoJSON data is None or empty")
        
        features = geojson.get('features', [])
        if not isinstance(features, list):
            raise ValueError("GeoJSON features must be a list")
        
        summary = {
            'total_features': len(features),
            'feature_types': defaultdict(int),
            'properties_summary': defaultdict(set),
            'coordinate_bounds': {'min_lat': float('inf'), 'max_lat': float('-inf'),
                                'min_lon': float('inf'), 'max_lon': float('-inf')},
            'feature_hashes': []
        }
        
        for i, feature in enumerate(features):
            if feature is None:
                print(f"Warning: Feature at index {i} is None, skipping...")
                continue
                
            # Count geometry types
            geometry = feature.get('geometry') if isinstance(feature, dict) else {}
            if geometry is None:
                geometry = {}
            geom_type = geometry.get('type', 'Unknown') if isinstance(geometry, dict) else 'Unknown'
            summary['feature_types'][geom_type] += 1
            
            # Collect property keys
            properties = feature.get('properties') if isinstance(feature, dict) else {}
            if properties is None:
                properties = {}
            if isinstance(properties, dict):
                for key in properties.keys():
                    if properties[key] is not None:
                        summary['properties_summary'][key].add(type(properties[key]).__name__)
            
            # Calculate bounds (simplified for points/first coordinate)
            try:
                coords = self._extract_coordinates(geometry)
                for coord in coords:
                    if coord and len(coord) >= 2 and coord[0] is not None and coord[1] is not None:
                        lon, lat = float(coord[0]), float(coord[1])
                        summary['coordinate_bounds']['min_lat'] = min(summary['coordinate_bounds']['min_lat'], lat)
                        summary['coordinate_bounds']['max_lat'] = max(summary['coordinate_bounds']['max_lat'], lat)
                        summary['coordinate_bounds']['min_lon'] = min(summary['coordinate_bounds']['min_lon'], lon)
                        summary['coordinate_bounds']['max_lon'] = max(summary['coordinate_bounds']['max_lon'], lon)
            except Exception as e:
                print(f"Warning: Error processing coordinates for feature {i}: {e}")
            
            # Create feature hash for exact matching
            try:
                feature_hash = self._create_feature_hash(feature)
                summary['feature_hashes'].append({
                    'index': i,
                    'hash': feature_hash,
                    'geometry_type': geom_type,
                    'properties': properties
                })
            except Exception as e:
                print(f"Warning: Error creating hash for feature {i}: {e}")
        
        # Convert sets to lists for JSON serialization
        summary['properties_summary'] = {k: list(v) for k, v in summary['properties_summary'].items()}
        
        # Handle case where no valid coordinates were found
        if summary['coordinate_bounds']['min_lat'] == float('inf'):
            summary['coordinate_bounds'] = {'min_lat': 0, 'max_lat': 0, 'min_lon': 0, 'max_lon': 0}
        
        return summary
    
    def _extract_coordinates(self, geometry: Dict[str, Any]) -> List[List[float]]:
        """Extract all coordinates from a geometry."""
        coords = []
        if not geometry or not isinstance(geometry, dict):
            return coords
            
        geom_type = geometry.get('type')
        coordinates = geometry.get('coordinates', [])
        
        if not coordinates:
            return coords
        
        try:
            if geom_type == 'Point':
                if coordinates and len(coordinates) >= 2:
                    coords.append(coordinates)
            elif geom_type in ['LineString', 'MultiPoint']:
                if isinstance(coordinates, list):
                    coords.extend([c for c in coordinates if c and len(c) >= 2])
            elif geom_type in ['Polygon', 'MultiLineString']:
                if isinstance(coordinates, list):
                    for ring in coordinates:
                        if isinstance(ring, list):
                            coords.extend([c for c in ring if c and len(c) >= 2])
            elif geom_type == 'MultiPolygon':
                if isinstance(coordinates, list):
                    for polygon in coordinates:
                        if isinstance(polygon, list):
                            for ring in polygon:
                                if isinstance(ring, list):
                                    coords.extend([c for c in ring if c and len(c) >= 2])
        except Exception as e:
            print(f"Warning: Error extracting coordinates from {geom_type}: {e}")
        
        return coords
    
    def _create_feature_hash(self, feature: Dict[str, Any]) -> str:
        """Create a hash for a feature to identify duplicates/matches."""
        if not feature or not isinstance(feature, dict):
            return hashlib.md5("".encode()).hexdigest()
            
        try:
            # Create a normalized representation
            normalized = {
                'geometry': feature.get('geometry'),
                'properties': feature.get('properties')
            }
            feature_str = json.dumps(normalized, sort_keys=True, separators=(',', ':'), default=str)
            return hashlib.md5(feature_str.encode()).hexdigest()
        except Exception as e:
            print(f"Warning: Error creating feature hash: {e}")
            return hashlib.md5(str(feature).encode()).hexdigest()
    
    def find_exact_matches(self, summary1: Dict[str, Any], summary2: Dict[str, Any]) -> Tuple[List, List, List]:
        """Find exact matches between features."""
        hashes1 = {item['hash']: item for item in summary1['feature_hashes']}
        hashes2 = {item['hash']: item for item in summary2['feature_hashes']}
        
        matches = []
        only_in_file1 = []
        only_in_file2 = []
        
        for hash_val, feature1 in hashes1.items():
            if hash_val in hashes2:
                matches.append({
                    'hash': hash_val,
                    'file1_index': feature1['index'],
                    'file2_index': hashes2[hash_val]['index']
                })
            else:
                only_in_file1.append(feature1)
        
        for hash_val, feature2 in hashes2.items():
            if hash_val not in hashes1:
                only_in_file2.append(feature2)
        
        return matches, only_in_file1, only_in_file2
    
    def compare_with_gemini(self, file1_path: str, file2_path: str) -> Dict[str, Any]:
        """Compare two GeoJSON files using Gemini API."""
        
        # Load and analyze files
        geojson1 = self.load_geojson(file1_path)
        geojson2 = self.load_geojson(file2_path)
        
        summary1 = self.extract_feature_summary(geojson1)
        summary2 = self.extract_feature_summary(geojson2)
        
        # Find exact matches
        matches, only_in_file1, only_in_file2 = self.find_exact_matches(summary1, summary2)
        
        # Prepare data for Gemini analysis
        comparison_data = {
            'file1_summary': {
                'total_features': summary1['total_features'],
                'feature_types': dict(summary1['feature_types']),
                'properties_summary': summary1['properties_summary'],
                'coordinate_bounds': summary1['coordinate_bounds']
            },
            'file2_summary': {
                'total_features': summary2['total_features'],
                'feature_types': dict(summary2['feature_types']),
                'properties_summary': summary2['properties_summary'],
                'coordinate_bounds': summary2['coordinate_bounds']
            },
            'exact_matches': len(matches),
            'features_only_in_file1': len(only_in_file1),
            'features_only_in_file2': len(only_in_file2),
            'sample_missing_from_file1': only_in_file2[:3],  # First 3 for analysis
            'sample_missing_from_file2': only_in_file1[:3]   # First 3 for analysis
        }
        
        # Create prompt for Gemini
        prompt = f"""
You are a GeoJSON comparison assistant. Given two GeoJSON file summaries and sample unmatched features, do the following:
1. Perform a deep feature-by-feature analysis, including:
   - Identifying *property differences* such as:
     - Key name changes
     - Value changes
     - Missing or extra keys
     - Type mismatches (e.g., string vs number)
   - Geometry changes:
     - Coordinate differences (even slight shifts)
     - Geometry type mismatches (e.g., Point vs Polygon)

2. Be very specific: check features as if line-by-line, and note *small textual changes*, such as:
   - "Name" → "name" (case sensitivity)
   - "Building Name" → "Building_Name" (formatting differences)
   - Coordinate precision changes (e.g., 19.123 → 19.1234)

3. Provide JSON-compatible examples from the unmatched samples.

---

## FILE COMPARISON SUMMARY:
{json.dumps(comparison_data, indent=2)}

---

## Output Format (strict):
### Feature Differences
- List field-by-field differences in properties and geometry
- Mention the index or hash of the mismatched feature

### Missing Features
- Which features are completely absent in the other file (based on hash or close match)

### Suggestions
- How to align schemas
- Rename keys or standardize formats if needed

Be extremely specific and avoid vague terms like "some keys are different" — enumerate all differences.
"""

        
        try:
            response = self.model.generate_content(prompt)
            gemini_analysis = response.text
        except Exception as e:
            gemini_analysis = f"Error calling Gemini API: {str(e)}"
        
        return {
            'summary_statistics': {
                'file1_features': summary1['total_features'],
                'file2_features': summary2['total_features'],
                'exact_matches': len(matches),
                'only_in_file1': len(only_in_file1),
                'only_in_file2': len(only_in_file2)
            },
            'detailed_differences': {
                'matches': matches,
                'missing_from_file2': only_in_file1,
                'missing_from_file1': only_in_file2
            },
            'gemini_analysis': gemini_analysis,
            'raw_summaries': {
                'file1': summary1,
                'file2': summary2
            }
        }
    
    def generate_report(self, comparison_result: Dict[str, Any], output_file: str = None):
        """Generate a formatted comparison report."""
        report = f"""
# GeoJSON Comparison Report

## Summary Statistics
- File 1 Features: {comparison_result['summary_statistics']['file1_features']}
- File 2 Features: {comparison_result['summary_statistics']['file2_features']}
- Exact Matches: {comparison_result['summary_statistics']['exact_matches']}
- Features only in File 1: {comparison_result['summary_statistics']['only_in_file1']}
- Features only in File 2: {comparison_result['summary_statistics']['only_in_file2']}

## AI Analysis
{comparison_result['gemini_analysis']}

## Detailed Technical Analysis
### Missing from File 2 (first 5):
"""
        
        for i, feature in enumerate(comparison_result['detailed_differences']['missing_from_file2'][:5]):
            report += f"\n{i+1}. Geometry: {feature.get('geometry_type', 'Unknown')}\n"
            report += f"   Properties: {list(feature.get('properties', {}).keys())}\n"
        
        report += "\n### Missing from File 1 (first 5):\n"
        for i, feature in enumerate(comparison_result['detailed_differences']['missing_from_file1'][:5]):
            report += f"\n{i+1}. Geometry: {feature.get('geometry_type', 'Unknown')}\n"
            report += f"   Properties: {list(feature.get('properties', {}).keys())}\n"
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report

# Usage example
def main():
    # Initialize comparator with your Gemini API key
    api_key = "AIzaSyDiQZDdfRX2cFqN5FuEX-wensjtzyzfq3E"
    
    # Check if files exist first
    import os
    file1 = 'file1.geojson'
    file2 = 'concatenated.geojson'
    
    if not os.path.exists(file1):
        print(f"Error: {file1} not found")
        return
    if not os.path.exists(file2):
        print(f"Error: {file2} not found")
        return
    
    print(f"Comparing {file1} and {file2}...")
    
    comparator = GeoJSONComparator(api_key)
    
    # Compare two GeoJSON files
    try:
        result = comparator.compare_with_gemini(file1, file2)
        
        # Generate and print report
        report = comparator.generate_report(result, 'comparison_report.txt')
        print(report)
        
        # Save detailed results as JSON
        with open('detailed_comparison.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
            
        print("\nComparison complete! Check 'comparison_report.txt' and 'detailed_comparison.json'")
            
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
