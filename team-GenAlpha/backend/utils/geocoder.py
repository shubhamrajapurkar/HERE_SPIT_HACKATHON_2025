import requests
import pandas as pd

HERE_API_KEY = "YOUR_HERE_API_KEY"  # Replace with your actual key

def get_lat_lng(query):
    """Get coordinates using HERE Geocoding API"""
    url = "https://geocode.search.hereapi.com/v1/geocode"
    params = {
        "q": query,
        "apiKey": HERE_API_KEY,
        "limit": 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('items'):
                position = data['items'][0]['position']
                return position['lat'], position['lng']
        return None, None
    except Exception as e:
        print(f"Geocoding error for {query}: {str(e)}")
        return None, None

def enrich_with_coordinates(df):
    """Add coordinates to DataFrame"""
    df['full_query'] = df['name'] + ' ' + df['address']
    df['latitude'], df['longitude'] = zip(*df['full_query'].apply(get_lat_lng))
    return df
