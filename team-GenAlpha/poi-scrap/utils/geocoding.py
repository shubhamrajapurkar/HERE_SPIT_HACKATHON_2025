# utils/geocoding.py
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
import logging

class GeocodingService:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="poi-scraper-v1.0")
        self.logger = logging.getLogger(__name__)
    
    def geocode_address(self, address, max_retries=3):
        """Get coordinates from address"""
        for attempt in range(max_retries):
            try:
                time.sleep(1)  # Rate limiting
                location = self.geolocator.geocode(address, timeout=10)
                if location:
                    return {
                        'latitude': location.latitude,
                        'longitude': location.longitude,
                        'formatted_address': location.address
                    }
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                self.logger.warning(f"Geocoding attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        return None
    
    def reverse_geocode(self, lat, lng, max_retries=3):
        """Get address from coordinates"""
        for attempt in range(max_retries):
            try:
                time.sleep(1)
                location = self.geolocator.reverse(f"{lat}, {lng}", timeout=10)
                if location:
                    return location.address
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                self.logger.warning(f"Reverse geocoding attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
        
        return None