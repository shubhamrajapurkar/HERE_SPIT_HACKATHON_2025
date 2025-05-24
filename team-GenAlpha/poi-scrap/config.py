import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Rate limiting
    REQUEST_DELAY = 2  # seconds between requests
    MAX_RETRIES = 3
    TIMEOUT = 30
    
    # File paths
    DATA_DIR = "data"
    LOGS_DIR = "logs"
    
    # Scraping settings
    MAX_RESULTS_PER_SITE = 100
    
    # Browser settings for Selenium
    HEADLESS = True
    CHROME_OPTIONS = [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--window-size=1920,1080",
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    ]
    
    # Categories to search for
    POI_CATEGORIES = [
        "restaurant", "shop", "mall", "office", "hospital", "school",
        "hotel", "bank", "atm", "gas station", "pharmacy", "gym"
    ]