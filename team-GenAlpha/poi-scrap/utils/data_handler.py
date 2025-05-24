import json
import os
import pandas as pd
from datetime import datetime
import logging

class DataHandler:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.setup_logging()
    
    def setup_logging(self):
        os.makedirs("logs", exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/scraper_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def save_data(self, data, filename, source=""):
        """Save data to JSON file with metadata"""
        filepath = os.path.join(self.data_dir, filename)
        
        output_data = {
            "metadata": {
                "source": source,
                "timestamp": datetime.now().isoformat(),
                "total_records": len(data),
                "scraper_version": "1.0"
            },
            "data": data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved {len(data)} records to {filepath}")
        return filepath
    
    def load_data(self, filename):
        """Load data from JSON file"""
        try:
           with open(filename, 'r') as f:
            return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
           self.logger.error(f"Error loading {filename}: {e}")
           return []
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def combine_all_data(self):
        """Combine data from all sources"""
        combined_data = []
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json') and filename != 'combined_data.json':
                data = self.load_data(filename)
                if data and 'data' in data:
                    combined_data.extend(data['data'])
        
        if combined_data:
            self.save_data(combined_data, 'combined_data.json', 'combined')
        
        return combined_data
