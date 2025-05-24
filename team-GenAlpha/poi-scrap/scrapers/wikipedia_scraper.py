# scrapers/wikipedia_scraper.py
import wikipedia
import re
import requests
from bs4 import BeautifulSoup
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import logging
from datetime import datetime
from typing import List, Dict, Optional

class WikipediaScraper:
    def __init__(self):
        self.logger = self._setup_logger()
        self.session = self._setup_session()
        wikipedia.set_lang("en")
        self.WIKI_API = "https://en.wikipedia.org/w/api.php"
        self.ROAD_KEYWORDS = ['road', 'highway', 'expressway', 'street', 'avenue', 'boulevard', 'marg']
        self.METRO_KEYWORDS = ['metro', 'station', 'subway', 'railway', 'train']
        self.DELAY = 1  # seconds between requests

    def _setup_logger(self):
        logger = logging.getLogger('WikipediaScraper')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def _setup_session(self):
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def search_location(self, location: str, categories: Optional[List[str]] = None) -> List[Dict]:
        """Main method to search for location data"""
        self.logger.info(f"Starting Wikipedia search for: {location}")
        
        results = []
        
        # Special handling for Mumbai
        if location.lower() == 'mumbai':
            results.extend(self._scrape_mumbai_specific_data())
        else:
            # Generic Wikipedia search
            search_terms = [
                f"List of roads in {location}",
                f"Transport in {location}",
                f"{location} Metro",
                f"Geography of {location}"
            ]
            
            for term in search_terms:
                try:
                    self.logger.info(f"Searching for: {term}")
                    page_results = self._search_wikipedia(term, location)
                    results.extend(page_results)
                except Exception as e:
                    self.logger.error(f"Error searching for {term}: {e}")
        
        return results

    def _scrape_mumbai_specific_data(self) -> List[Dict]:
        """Specialized scraper for Mumbai transport data"""
        mumbai_data = []
        
        # Scrape Mumbai roads
        try:
            roads = self._scrape_mumbai_roads()
            mumbai_data.extend(roads)
            self.logger.info(f"Found {len(roads)} Mumbai roads")
        except Exception as e:
            self.logger.error(f"Failed to scrape Mumbai roads: {e}")
        
        # Scrape Mumbai metro stations
        try:
            metro_stations = self._scrape_mumbai_metro()
            mumbai_data.extend(metro_stations)
            self.logger.info(f"Found {len(metro_stations)} Mumbai metro stations")
        except Exception as e:
            self.logger.error(f"Failed to scrape Mumbai metro: {e}")
        
        return mumbai_data

    def _scrape_mumbai_roads(self) -> List[Dict]:
        """Scrape Mumbai roads data with addresses"""
        roads = []
        page_title = "List of roads in Mumbai"
        
        try:
            page = wikipedia.page(page_title)
            soup = BeautifulSoup(page.html(), 'html.parser')
            
            # Find all list items containing road information
            for li in soup.select('li'):
                text = li.get_text().strip()
                if any(keyword in text.lower() for keyword in self.ROAD_KEYWORDS):
                    road_data = self._process_road_text(text, "Mumbai")
                    if road_data:
                        roads.append(road_data)
            
            # Also check tables for road data
            for table in soup.find_all('table', {'class': 'wikitable'}):
                for row in table.find_all('tr')[1:]:  # Skip header
                    cols = row.find_all('td')
                    if len(cols) >= 1:
                        road_data = self._process_road_text(cols[0].get_text().strip(), "Mumbai")
                        if road_data:
                            roads.append(road_data)
            
            # Get details for major roads
            major_roads = [
                "Western Express Highway",
                "Eastern Express Highway",
                "Linking Road, Mumbai",
                "Marine Drive, Mumbai",
                "S.V. Road, Mumbai",
                "L.B.S. Marg, Mumbai",
                "Dr. Annie Besant Road, Mumbai",
                "P.D'Mello Road, Mumbai"
            ]
            
            for road in major_roads:
                try:
                    road_details = self._get_road_details(road, "Mumbai")
                    if road_details:
                        roads.append(road_details)
                except Exception as e:
                    self.logger.warning(f"Couldn't get details for {road}: {e}")
            
        except Exception as e:
            self.logger.error(f"Error scraping Mumbai roads: {e}")
        
        return roads

    def _scrape_mumbai_metro(self) -> List[Dict]:
        """Scrape Mumbai metro stations data with more details"""
        stations = []
        page_titles = [
            "List of Mumbai Metro stations",
            "Mumbai Metro",
            "Mumbai Monorail"
        ]
        
        for page_title in page_titles:
            try:
                page = wikipedia.page(page_title)
                soup = BeautifulSoup(page.html(), 'html.parser')
                
                # Find station tables
                for table in soup.find_all('table', {'class': 'wikitable'}):
                    headers = [th.get_text().strip().lower() for th in table.find_all('th')]
                    
                    if 'station' in headers or any(h in ['name', 'station name'] for h in headers):
                        for row in table.find_all('tr')[1:]:  # Skip header
                            cols = row.find_all('td')
                            if len(cols) >= 1:  # At least station name
                                station_name = cols[0].get_text().strip()
                                station_data = {
                                    'name': station_name,
                                    'category': 'metro_station',
                                    'source': 'wikipedia',
                                    'scrape_time': datetime.now().isoformat()
                                }
                                
                                # Try to get line information
                                if len(cols) >= 2:
                                    station_data['line'] = cols[1].get_text().strip()
                                
                                # Try to get interchange information
                                if len(cols) >= 3:
                                    station_data['interchange'] = cols[2].get_text().strip()
                                
                                # Get coordinates and address
                                coords = self._get_coordinates_for_station(station_name)
                                if coords:
                                    station_data.update(coords)
                                
                                # Get address if no coordinates
                                if 'latitude' not in station_data:
                                    try:
                                        station_page = wikipedia.page(station_name)
                                        address = self._extract_address(station_page.content, "Mumbai")
                                        if address:
                                            station_data['address'] = address
                                    except:
                                        pass
                                
                                stations.append(station_data)
                
            except Exception as e:
                self.logger.error(f"Error scraping {page_title}: {e}")
        
        return stations

    def _search_wikipedia(self, search_term: str, location: str) -> List[Dict]:
        """Generic Wikipedia search method"""
        results = []
        
        try:
            search_results = wikipedia.search(search_term, results=10)
            
            for title in search_results:
                try:
                    page_data = self._extract_page_data(title, location)
                    if page_data:
                        results.append(page_data)
                except Exception as e:
                    self.logger.warning(f"Error processing {title}: {e}")
        
        except wikipedia.exceptions.DisambiguationError as e:
            self.logger.info(f"Disambiguation needed for {search_term}, trying first option")
            try:
                page_data = self._extract_page_data(e.options[0], location)
                if page_data:
                    results.append(page_data)
            except:
                pass
        
        return results

    def _extract_page_data(self, title: str, location: str) -> Optional[Dict]:
        """Extract data from a Wikipedia page"""
        try:
            page = wikipedia.page(title)
            soup = BeautifulSoup(page.html(), 'html.parser')
            
            data = {
                'name': page.title,
                'category': self._determine_category(page.content, title),
                'description': page.summary[:500] if page.summary else '',
                'url': page.url,
                'source': 'wikipedia',
                'scrape_time': datetime.now().isoformat()
            }
            
            # Get coordinates
            coords = self._extract_coordinates(page, soup)
            if coords:
                data.update(coords)
            
            # Get address if no coordinates
            if 'latitude' not in data:
                address = self._extract_address(page.content, location)
                if address:
                    data['address'] = address
            
            # Add features based on content
            data['features'] = self._extract_features(page.content)
            
            return data
        
        except wikipedia.exceptions.PageError:
            return None
        except Exception as e:
            self.logger.warning(f"Error extracting from {title}: {e}")
            return None

    def _extract_coordinates(self, page, soup) -> Optional[Dict]:
        """Extract coordinates using multiple methods"""
        try:
            # Method 1: Direct coordinates attribute
            if hasattr(page, 'coordinates') and page.coordinates:
                return {
                    'latitude': float(page.coordinates[0]),
                    'longitude': float(page.coordinates[1])
                }
            
            # Method 2: From geo span
            geo_span = soup.find('span', {'class': 'geo'})
            if geo_span:
                lat, lon = geo_span.get_text().split(';')
                return {
                    'latitude': float(lat.strip()),
                    'longitude': float(lon.strip())
                }
            
            # Method 3: From geohack URL
            geohack_link = soup.find('a', href=lambda x: x and 'geohack' in x)
            if geohack_link:
                match = re.search(r'params=([\d\.]+)_([NS])_([\d\.]+)_([EW])', geohack_link['href'])
                if match:
                    lat = float(match.group(1)) * (-1 if match.group(2) == 'S' else 1)
                    lon = float(match.group(3)) * (-1 if match.group(4) == 'W' else 1)
                    return {
                        'latitude': lat,
                        'longitude': lon
                    }
            
            return None
        except Exception as e:
            self.logger.warning(f"Coordinate extraction failed: {e}")
            return None

    def _determine_category(self, content: str, title: str) -> str:
        """Determine the category of the POI"""
        content_lower = content.lower()
        title_lower = title.lower()
        
        if any(keyword in title_lower for keyword in self.METRO_KEYWORDS):
            return 'metro_station'
        elif any(keyword in title_lower for keyword in self.ROAD_KEYWORDS):
            return 'road'
        elif 'airport' in title_lower:
            return 'airport'
        elif 'hospital' in content_lower:
            return 'hospital'
        elif 'school' in content_lower or 'university' in content_lower:
            return 'education'
        elif 'mall' in content_lower or 'shopping' in content_lower:
            return 'shopping'
        elif 'park' in content_lower:
            return 'park'
        else:
            return 'point_of_interest'

    def _extract_address(self, content: str, location: str) -> Optional[str]:
        """Extract address from content with improved patterns"""
        patterns = [
            r'located (?:at|in|on) ([^.,]+(?:' + '|'.join(self.ROAD_KEYWORDS) + r')[^.,]*)',
            r'address[:\s]+([^.,]+)',
            r'situated (?:at|in|on) ([^.,]+)',
            r'found (?:at|in|on) ([^.,]+)',
            r'along ([^.,]+)',
            r'near ([^.,]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                address = match.group(1).strip()
                # Clean the address
                address = re.sub(r'\[.*?\]', '', address)  # Remove citations
                address = re.sub(r'\s+', ' ', address)  # Remove extra spaces
                if location.lower() not in address.lower():
                    address += f", {location}"
                return address
        
        return None

    def _extract_features(self, content: str) -> List[str]:
        """Extract features from content"""
        features = []
        content_lower = content.lower()
        
        if 'parking' in content_lower:
            features.append('parking')
        if 'wheelchair' in content_lower:
            features.append('wheelchair_accessible')
        if 'wifi' in content_lower:
            features.append('wifi')
        if 'restaurant' in content_lower or 'food' in content_lower:
            features.append('food')
        if 'shop' in content_lower or 'store' in content_lower:
            features.append('shopping')
        
        return features

    def _process_road_text(self, text: str, location: str) -> Optional[Dict]:
        """Process road text into structured data with address"""
        try:
            # Clean the text
            text = re.sub(r'\[.*?\]', '', text)  # Remove citations
            text = text.split('(')[0].strip()  # Remove parentheses content
            
            if len(text) < 5 or not any(keyword in text.lower() for keyword in self.ROAD_KEYWORDS):
                return None
            
            road_data = {
                'name': text,
                'category': 'road',
                'source': 'wikipedia',
                'scrape_time': datetime.now().isoformat(),
                'address': f"{text}, {location}"  # Basic address format
            }
            
            # Try to get more detailed information if possible
            try:
                page = wikipedia.page(text)
                soup = BeautifulSoup(page.html(), 'html.parser')
                
                # Get coordinates if available
                coords = self._extract_coordinates(page, soup)
                if coords:
                    road_data.update(coords)
                
                # Extract better address from content
                address = self._extract_address(page.content, location)
                if address:
                    road_data['address'] = address
                
                # Add description if available
                if page.summary:
                    road_data['description'] = page.summary[:500]
                
            except:
                pass
            
            return road_data
        except Exception as e:
            self.logger.warning(f"Error processing road text: {e}")
            return None

    def _get_road_details(self, road_name: str, location: str) -> Optional[Dict]:
        """Get detailed information about a road including address"""
        try:
            page = wikipedia.page(road_name)
            soup = BeautifulSoup(page.html(), 'html.parser')
            
            details = {
                'name': page.title,
                'category': 'road',
                'description': page.summary[:500] if page.summary else '',
                'url': page.url,
                'source': 'wikipedia',
                'scrape_time': datetime.now().isoformat(),
                'address': f"{page.title}, {location}"  # Default address
            }
            
            # Extract from infobox
            infobox = soup.find('table', {'class': 'infobox'})
            if infobox:
                for row in infobox.find_all('tr'):
                    header = row.find('th')
                    data = row.find('td')
                    if header and data:
                        header_text = header.get_text().strip().lower()
                        data_text = data.get_text().strip()
                        
                        if 'length' in header_text:
                            details['length'] = data_text
                        elif 'maintained' in header_text:
                            details['maintained_by'] = data_text
                        elif 'terminus' in header_text:
                            details['terminals'] = [t.strip() for t in data_text.split(' to ')]
                        elif 'location' in header_text:
                            details['address'] = data_text + f", {location}"
            
            # Get coordinates
            coords = self._extract_coordinates(page, soup)
            if coords:
                details.update(coords)
            
            # Get better address from content if not found in infobox
            if 'address' not in details or details['address'] == f"{page.title}, {location}":
                address = self._extract_address(page.content, location)
                if address:
                    details['address'] = address
            
            return details
        except:
            return None

    def _get_coordinates_for_station(self, station_name: str) -> Optional[Dict]:
        """Get coordinates for a metro station with more robust handling"""
        try:
            page = wikipedia.page(station_name)
            soup = BeautifulSoup(page.html(), 'html.parser')
            
            # First try standard coordinate extraction
            coords = self._extract_coordinates(page, soup)
            if coords:
                return coords
            
            # If not found, try to find in infobox
            infobox = soup.find('table', {'class': 'infobox'})
            if infobox:
                # Look for coordinates in infobox rows
                for row in infobox.find_all('tr'):
                    if 'coordinates' in row.get_text().lower():
                        coords = self._extract_coordinates(page, row)
                        if coords:
                            return coords
                
                # Look for map links
                for link in infobox.find_all('a', href=True):
                    if 'geohack' in link['href']:
                        match = re.search(r'params=([\d\.]+)_([NS])_([\d\.]+)_([EW])', link['href'])
                        if match:
                            lat = float(match.group(1)) * (-1 if match.group(2) == 'S' else 1)
                            lon = float(match.group(3)) * (-1 if match.group(4) == 'W' else 1)
                            return {
                                'latitude': lat,
                                'longitude': lon
                            }
            
            return None
        except Exception as e:
            self.logger.warning(f"Failed to get coordinates for {station_name}: {e}")
            return None