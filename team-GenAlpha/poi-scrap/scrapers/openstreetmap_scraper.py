# scrapers/openstreetmap_scraper.py
from .base_scraper import BaseScraper
import overpy

class OpenStreetMapScraper(BaseScraper):
    def __init__(self):
        super().__init__(delay=1)
        self.api = overpy.Overpass()
    
    def search_location(self, location, categories=None):
        """Search for POIs in a location using OpenStreetMap Overpass API"""
        self.logger.info(f"Scraping OpenStreetMap for POIs in {location}")
        
        # Get bounding box for the location
        bbox = self._get_location_bbox(location)
        if not bbox:
            self.logger.error(f"Could not find bounding box for {location}")
            return []
        
        all_results = []
        
        # Define OSM queries for different POI types
        osm_queries = {
            'restaurant': ['amenity=restaurant', 'amenity=cafe', 'amenity=fast_food'],
            'shop': ['shop=*'],
            'mall': ['shop=mall', 'amenity=marketplace'],
            'office': ['office=*', 'building=office'],
            'hotel': ['tourism=hotel', 'tourism=guest_house'],
            'hospital': ['amenity=hospital', 'amenity=clinic'],
            'school': ['amenity=school', 'amenity=university', 'amenity=college']
        }
        
        for category, queries in osm_queries.items():
            for query in queries:
                try:
                    results = self._query_overpass(query, bbox, category)
                    all_results.extend(results)
                except Exception as e:
                    self.logger.error(f"Error querying {query}: {e}")
        
        return all_results
    
    def _get_location_bbox(self, location):
        """Get bounding box for location using Nominatim"""
        try:
            from utils.geocoding import GeocodingService
            geocoding = GeocodingService()
            
            # Get location coordinates
            coords = geocoding.geolocator.geocode(location, timeout=10)
            if coords:
                # Create a bounding box around the location (roughly 10km radius)
                lat, lon = coords.latitude, coords.longitude
                offset = 0.05  # roughly 5km
                
                return {
                    'south': lat - offset,
                    'west': lon - offset,
                    'north': lat + offset,
                    'east': lon + offset
                }
        except Exception as e:
            self.logger.error(f"Error getting bbox for {location}: {e}")
        
        return None
    
    def _query_overpass(self, query, bbox, category):
        """Query Overpass API for POIs"""
        overpass_query = f"""
        [out:json][timeout:25];
        (
          node[{query}]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
          way[{query}]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
          relation[{query}]({bbox['south']},{bbox['west']},{bbox['north']},{bbox['east']});
        );
        out center meta;
        """
        
        try:
            result = self.api.query(overpass_query)
            return self._parse_overpass_result(result, category)
        except Exception as e:
            self.logger.error(f"Overpass query error: {e}")
            return []
    
    def _parse_overpass_result(self, result, category):
        """Parse Overpass API result"""
        pois = []
        
        # Process nodes
        for node in result.nodes:
            poi_data = self._extract_osm_data(node, category)
            if poi_data:
                pois.append(self.standardize_poi_data(poi_data))
        
        # Process ways (buildings, areas)
        for way in result.ways:
            poi_data = self._extract_osm_data(way, category)
            if poi_data:
                pois.append(self.standardize_poi_data(poi_data))
        
        # Process relations
        for relation in result.relations:
            poi_data = self._extract_osm_data(relation, category)
            if poi_data:
                pois.append(self.standardize_poi_data(poi_data))
        
        return pois
    
    def _extract_osm_data(self, osm_element, category):
        """Extract POI data from OSM element"""
        tags = osm_element.tags
        
        # Skip if no name
        if 'name' not in tags:
            return None
        
        data = {
            'name': tags.get('name', ''),
            'category': category
        }
        
        # Coordinates
        if hasattr(osm_element, 'lat') and hasattr(osm_element, 'lon'):
            data['latitude'] = float(osm_element.lat)
            data['longitude'] = float(osm_element.lon)
        elif hasattr(osm_element, 'center_lat') and hasattr(osm_element, 'center_lon'):
            data['latitude'] = float(osm_element.center_lat)
            data['longitude'] = float(osm_element.center_lon)
        
        # Address components
        address_parts = []
        for addr_key in ['addr:housenumber', 'addr:street', 'addr:city', 'addr:postcode']:
            if addr_key in tags:
                address_parts.append(tags[addr_key])
        
        if address_parts:
            data['address'] = ', '.join(address_parts)
        
        # Contact information
        if 'phone' in tags:
            data['phone'] = tags['phone']
        if 'website' in tags:
            data['website'] = tags['website']
        
        # Features
        features = []
        feature_tags = {
            'wifi': 'internet_access',
            'parking': 'parking',
            'wheelchair': 'wheelchair',
            'outdoor_seating': 'outdoor_seating'
        }
        
        for feature, tag in feature_tags.items():
            if tag in tags and tags[tag] in ['yes', 'true', '1']:
                features.append(feature)
        
        data['features'] = features
        
        return data
