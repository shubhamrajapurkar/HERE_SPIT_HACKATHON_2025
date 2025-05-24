import geopandas as gpd
import pandas as pd
import os

# Load the shapefile
gdf = gpd.read_file(r"Streets\Streets.shp")

# Load the coordinates from the CSV
csv_df = pd.read_csv("merged_nodes_og.csv")

# Create a set of (X, Y) tuples from the CSV for fast lookup, rounded to avoid float mismatch
coord_set = set(zip(csv_df["X coord"].round(5), csv_df["Y coord"].round(5)))

# Function to check if any coordinate in the geometry exists in coord_set
def check_line_string_for_coords(geometry):
    if geometry.geom_type == 'LineString':
        for x, y in geometry.coords:
            if (round(x, 5), round(y, 5)) in coord_set:
                return "Yes"
    elif geometry.geom_type == 'MultiLineString':
        for line in geometry.geoms:
            for x, y in line.coords:
                if (round(x, 5), round(y, 5)) in coord_set:
                    return "Yes"
    return "No"

# Apply the function and create a new column
gdf["Matched"] = gdf["geometry"].apply(check_line_string_for_coords)

# Fix: Convert LINK_ID to string to avoid integer overflow error when writing to .shp
if "LINK_ID" in gdf.columns:
    gdf["LINK_ID"] = gdf["LINK_ID"].astype(str)

# Save the updated shapefile
os.makedirs("output", exist_ok=True)
gdf.to_file("output/updated_file.shp")

# Filter and print only the rows where Matched == "Yes"
matched_rows = gdf[gdf["Matched"] == "Yes"]
print(matched_rows)
