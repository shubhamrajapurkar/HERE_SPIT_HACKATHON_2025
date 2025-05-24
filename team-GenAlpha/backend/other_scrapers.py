# In other_scrapers.py
from flask import Blueprint
from scrapers.justdial_scraper import JustDialScraper
from scrapers.wikipedia_scraper import WikipediaScraper

other_scrapers_bp = Blueprint('scraper', __name__)

@other_scrapers_bp.route('/justdial/<location>', methods=['GET'])
def scrape_justdial(location):
    scraper = JustDialScraper()
    results = scraper.search_location(location)
    return {'data': results}

@other_scrapers_bp.route('/wikipedia/<location>', methods=['GET'])
def scrape_wikipedia(location):
    scraper = WikipediaScraper()
    results = scraper.search_location(location)
    return {'data': results}