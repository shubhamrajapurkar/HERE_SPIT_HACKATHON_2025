
import argparse
import sys
import os
from config import Config
from utils.data_handler import DataHandler
from scrapers.justdial_scraper import JustDialScraper
from scrapers.wikipedia_scraper import WikipediaScraper
from scrapers.openstreetmap_scraper import OpenStreetMapScraper

def setup_directories():
    """Create necessary directories"""
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.LOGS_DIR, exist_ok=True)

def run_justdial_scraper(location):
    """Run JustDial scraper only"""
    print(f"🔍 Starting JustDial scraper for: {location}")
    
    scraper = JustDialScraper()
    data_handler = DataHandler()
    
    try:
        results = scraper.search_location(location, Config.POI_CATEGORIES)
        if results:
            data_handler.save_data(results, 'justdial_data.json', 'JustDial')
            print(f"✅ JustDial: Scraped {len(results)} POIs")
        else:
            print("❌ JustDial: No results found")
    except Exception as e:
        print(f"❌ JustDial scraper failed: {e}")

def run_wikipedia_scraper(location):
    """Run Wikipedia scraper only"""
    print(f"🔍 Starting Wikipedia scraper for: {location}")
    
    scraper = WikipediaScraper()
    data_handler = DataHandler()
    
    try:
        results = scraper.search_location(location)
        if results:
            data_handler.save_data(results, 'wikipedia_data.json', 'Wikipedia')
            print(f"✅ Wikipedia: Scraped {len(results)} POIs")
        else:
            print("❌ Wikipedia: No results found")
    except Exception as e:
        print(f"❌ Wikipedia scraper failed: {e}")

def run_osm_scraper(location):
    """Run OpenStreetMap scraper only"""
    print(f"🔍 Starting OpenStreetMap scraper for: {location}")
    
    scraper = OpenStreetMapScraper()
    data_handler = DataHandler()
    
    try:
        results = scraper.search_location(location)
        if results:
            data_handler.save_data(results, 'osm_data.json', 'OpenStreetMap')
            print(f"✅ OpenStreetMap: Scraped {len(results)} POIs")
        else:
            print("❌ OpenStreetMap: No results found")
    except Exception as e:
        print(f"❌ OpenStreetMap scraper failed: {e}")

def run_all_scrapers(location):
    """Run all scrapers"""
    print(f"🚀 Starting all scrapers for: {location}")
    
    data_handler = DataHandler()
    total_results = 0
    
    # Run each scraper
    scrapers = [
        # (JustDialScraper, 'justdial_data.json', 'JustDial'),
        (WikipediaScraper, 'wikipedia_data.json', 'Wikipedia'),
        (OpenStreetMapScraper, 'osm_data.json', 'OpenStreetMap')
    ]
    
    for ScraperClass, filename, source in scrapers:
        try:
            print(f"\n🔍 Running {source} scraper...")
            scraper = ScraperClass()
            results = scraper.search_location(location, Config.POI_CATEGORIES if source == 'JustDial' else None)
            
            if results:
                data_handler.save_data(results, filename, source)
                total_results += len(results)
                print(f"✅ {source}: Scraped {len(results)} POIs")
            else:
                print(f"❌ {source}: No results found")
                
        except Exception as e:
            print(f"❌ {source} scraper failed: {e}")
    
    # Combine all data
    if total_results > 0:
        print(f"\n📊 Combining all data...")
        combined_data = data_handler.combine_all_data()
        print(f"✅ Combined {len(combined_data)} POIs total")
        print(f"📁 Data saved in '{Config.DATA_DIR}' directory")
    else:
        print("❌ No data scraped from any source")

def main():
    parser = argparse.ArgumentParser(description='POI Scraper - Extract Points of Interest from multiple sources')
    parser.add_argument('location', help='Location to search for POIs (e.g., "New York", "Mumbai")')
    parser.add_argument('--source', choices=['justdial', 'wikipedia', 'osm', 'all'], 
                       default='all', help='Source to scrape from (default: all)')
    
    args = parser.parse_args()
    
    if not args.location:
        print("❌ Please provide a location to search")
        sys.exit(1)
    
    # Setup
    setup_directories()
    
    print("=" * 60)
    print("🗺️  POI SCRAPER")
    print("=" * 60)
    
    # Run appropriate scraper
    # if args.source == 'justdial':
    #     run_justdial_scraper(args.location)
    if args.source == 'wikipedia':
        run_wikipedia_scraper(args.location)
    elif args.source == 'osm':
        run_osm_scraper(args.location)
    else:
        run_all_scrapers(args.location)
    
    print("\n" + "=" * 60)
    print("🎉 Scraping completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()