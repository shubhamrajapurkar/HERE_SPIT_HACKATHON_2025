def scraped_data_to_geojson(scraped_data):
    """Convert enriched scraped data to GeoJSON"""
    features = []
    
    for item in scraped_data:
        if item['latitude'] and item['longitude']:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [item['longitude'], item['latitude']]
                },
                "properties": {
                    "name": item['name'],
                    "address": item['address'],
                    "category": item['category'],
                    "phone": item['phone'],
                    "rating": item['rating'],
                    "source": item['source'],
                    "website": item['website'],
                    "scraped_at": item['scraped_at']
                }
            }
            features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features
    }
