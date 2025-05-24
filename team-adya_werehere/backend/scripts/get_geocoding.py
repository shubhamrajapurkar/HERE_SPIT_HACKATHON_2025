import json
import os
import time

from dotenv import load_dotenv
from opencage.geocoder import OpenCageGeocode
from tqdm import tqdm

load_dotenv()


with open("../data/restaurant_review.json") as f:
    data = json.load(f)

with OpenCageGeocode(os.environ.get("OC_API_KEY")) as geocoder:
    for restaurant in tqdm(data, desc="Geocoding locations"):
        if restaurant.get("location") and restaurant["location"]:
            results = geocoder.geocode(restaurant["location"])
            coords: None | dict[str, str] = None
            if results and len(results):
                coords = {
                    "lat": results[0]["geometry"]["lat"],  # type: ignore
                    "lng": results[0]["geometry"]["lng"],  # type: ignore
                }
                print(coords)

            if coords:
                restaurant["coordinates"] = coords

            time.sleep(0.01)

with open("../data/restaurant_review.json", "w") as f:
    json.dump(data, f, indent=2)

print("Geocoding completed and saved to file")
