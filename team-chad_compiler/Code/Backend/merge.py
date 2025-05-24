import pandas as pd

# Load the CSV files
nodes_table = pd.read_csv('nodes_table.csv') 
nodes = pd.read_csv('circle_nodes_output.csv')

# Merge the two dataframes on 'Node'
merged = pd.merge(nodes, nodes_table, on='Node', how='inner')

# Save the result to a new CSV file
merged.to_csv('merged_nodes.csv', index=False)

print("Merged file 'merged_nodes.csv' created successfully.")
