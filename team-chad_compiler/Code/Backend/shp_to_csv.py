import geopandas as gpd
import pandas as pd

# Load shapefile
gdf = gpd.read_file(r"Streets\Streets.shp")

# Dictionary for unique node coordinates → ID
nodes = {}
edges = []
node_counter = 1

for geom in gdf.geometry:
    if geom.geom_type == 'LineString':
        coords = list(geom.coords)
    elif geom.geom_type == 'MultiLineString':
        coords = [pt for linestring in geom.geoms for pt in linestring.coords]
    else:
        continue

    previous_node = None
    for coord in coords:
        coord = tuple(round(c, 8) for c in coord)  # round to avoid float issues
        if coord not in nodes:
            nodes[coord] = node_counter
            node_counter += 1

        current_node = nodes[coord]

        if previous_node is not None:
            edges.append((previous_node, current_node))

        previous_node = current_node

# Create Node DataFrame
node_df = pd.DataFrame([
    {'Node': node_id, 'X coord': coord[0], 'Y coord': coord[1]}
    for coord, node_id in nodes.items()
])

# Create Edge DataFrame
edge_df = pd.DataFrame(edges, columns=['Node1', 'Node2']).drop_duplicates()

# Optional: Sort nodes by ID
node_df.sort_values(by='Node', inplace=True)

# --- Save to CSV files ---
node_df.to_csv("nodes_table.csv", index=False)
edge_df.to_csv("edges_table.csv", index=False)

print("✔️ Nodes and edges tables saved as 'nodes_table.csv' and 'edges_table.csv'")
