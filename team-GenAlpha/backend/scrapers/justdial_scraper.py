# scrapers/justdial_scraper.py
from .base_scraper import BaseScraper
from bs4 import BeautifulSoup
import re
import urllib.parse

class JustDialScraper(BaseScraper):
    def __init__(self):
        super().__init__(delay=3)  # Higher delay for JustDial
        self.base_url = "https://www.justdial.com"
    
    def search_location(self, location, categories=None):
        """Search for POIs in a location on JustDial"""
        if not categories:
            categories = ["restaurants", "shops", "malls", "offices"]
        
        all_results = []
        
        for category in categories:
            self.logger.info(f"Scraping JustDial for {category} in {location}")
            try:
                results = self._search_category(location, category)
                all_results.extend(results)
            except Exception as e:
                self.logger.error(f"Error scraping {category}: {e}")
        
        return all_results
    
    def _search_category(self, location, category):
        """Search for a specific category"""
        # Format location and category for JustDial URL
        location_formatted = location.replace(' ', '-').lower()
        category_formatted = category.replace(' ', '-').lower()
        
        search_url = f"{self.base_url}/{location_formatted}/{category_formatted}"
        
        try:
            response = self.make_request(search_url)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            return self._parse_search_results(soup, category)
        except Exception as e:
            self.logger.error(f"Error searching {category} in {location}: {e}")
            return []
    
    def _parse_search_results(self, soup, category):
        """Parse search results from JustDial page"""
        results = []
        
        # Find listing containers (JustDial structure may vary)
        listings = soup.find_all('div', class_=re.compile('resultbox|listingbox'))
        
        for listing in listings[:50]:  # Limit results
            try:
                poi_data = self._extract_poi_data(listing, category)
                if poi_data:
                    results.append(self.standardize_poi_data(poi_data))
            except Exception as e:
                self.logger.warning(f"Error parsing listing: {e}")
        
        return results
    
    def _extract_poi_data(self, listing, category):
        """Extract POI data from a listing"""
        data = {}
        
        # Name
        name_elem = listing.find(['h2', 'h3', 'a'], class_=re.compile('fn|title'))
        if name_elem:
            data['name'] = name_elem.get_text(strip=True)
        
        # Address
        address_elem = listing.find(['span', 'div'], class_=re.compile('mrehover|address'))
        if address_elem:
            data['address'] = address_elem.get_text(strip=True)
        
        # Phone
        phone_elem = listing.find('span', class_=re.compile('mobilesv|phone'))
        if phone_elem:
            data['phone'] = phone_elem.get_text(strip=True)
        
        # Rating
        rating_elem = listing.find(['span', 'div'], class_=re.compile('rating|star'))
        if rating_elem:
            rating_text = rating_elem.get_text(strip=True)
            rating_match = re.search(r'(\d+\.?\d*)', rating_text)
            if rating_match:
                data['rating'] = float(rating_match.group(1))
        
        data['category'] = category
        
        return data if data.get('name') else None