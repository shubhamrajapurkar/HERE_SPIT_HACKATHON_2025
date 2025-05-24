import json

from tqdm import tqdm

with open("../data/restaurant_review.json") as f:
    data = json.load(f)

for dp in tqdm(data, desc="Modify Json"):
    lat, lgn = 0, 0
    if dp.get("latitude"):
        lat = dp["latitude"]
        del dp["latitude"]
    if dp.get("longitude"):
        lgn = dp["longitude"]
        del dp["longitude"]

    coords = {"lat": lat, "lng": lgn}
    dp["coords"] = coords

with open("../data/restaurant_review_copy.json", "w") as f:
    json.dump(data, f, indent=2)
