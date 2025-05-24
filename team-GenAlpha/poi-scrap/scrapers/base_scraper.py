# scrapers/base_scraper.py
import time
import logging
import requests
from fake_useragent import UserAgent
from abc import ABC, abstractmethod

class BaseScraper(ABC):
    def __init__(self, delay=2):
        self.delay = delay
        self.session = requests.Session()
        self.ua = UserAgent()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.setup_session()
    
    def setup_session(self):
        """Setup session with headers"""
        self.session.headers.update({
            'User-Agent': self.ua.random,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
    
    def make_request(self, url, max_retries=3):
        """Make HTTP request with retry logic"""
        for attempt in range(max_retries):
            try:
                time.sleep(self.delay)
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response
            except requests.RequestException as e:
                self.logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
    
    @abstractmethod
    def search_location(self, location, categories=None):
        """Abstract method to be implemented by each scraper"""
        pass
    
    def standardize_poi_data(self, raw_data):
        """Standardize POI data format"""
        return {
            'name': raw_data.get('name', ''),
            'address': raw_data.get('address', ''),
            'latitude': raw_data.get('latitude'),
            'longitude': raw_data.get('longitude'),
            'category': raw_data.get('category', ''),
            'phone': raw_data.get('phone', ''),
            'website': raw_data.get('website', ''),
            'rating': raw_data.get('rating'),
            'features': raw_data.get('features', []),
            'source': self.__class__.__name__,
            'scraped_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }