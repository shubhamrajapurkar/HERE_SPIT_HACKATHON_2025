import pandas as pd
import numpy as np
import math
from collections import defaultdict
import csv

class Node:
    def __init__(self, idx, x, y):
        self.idx = idx
        self.x = x
        self.y = y

class CircleDetector:
    def __init__(self, nodes, adj_list):
        self.nodes = nodes
        self.adj = adj_list
        self.visited = set()
        self.cur_ans = []  # Current path as edges
        self.fans = []  # All detected cycles
        self.start_node = None  # Global start node
    
    def calculate_angle(self, from_node, to_node):
        """Calculate angle between two points in degrees"""
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        angle = math.atan2(dy, dx) * 180.0 / math.pi
        return angle if angle >= 0 else angle + 360  # Normalize to [0, 360)
    
    def angle_difference(self, angle1, angle2):
        """Normalize angle difference to [-180, 180] - returns signed difference"""
        diff = angle2 - angle1
        while diff > 180:
            diff -= 360
        while diff < -180:
            diff += 360
        return diff  # Return signed difference, not absolute (like C++ version)
    
    def dfs(self, current_node, prev_node, prev_angle, prev_diff, depth):
        """DFS to detect circles with depth limiting and enhanced angle logic"""
        # Limit recursion depth
        if depth >= 10000:
            return
        
        # Check if we've found a cycle - UPDATED CONDITION
        if current_node in self.visited:
            # Only accept the cycle if we returned to the start node
            if current_node == self.start_node and len(self.cur_ans) > 0:
                self.fans.append(self.cur_ans.copy())
            return
        
        self.visited.add(current_node)
        
        # Get all adjacent nodes except the previous node
        candidates = []
        
        for next_node in self.adj[current_node]:
            if next_node == prev_node:
                continue  # Skip parent node
            
            angle = self.calculate_angle(self.nodes[current_node], self.nodes[next_node])
            diff = self.angle_difference(prev_angle, angle)
            
            candidates.append((next_node, diff))
        
        for next_node, diff in candidates:
            # Check if angle difference is within range
            if abs(diff) >= 0.000001 and abs(diff) <= 50:
                # Check if signs are the same (both positive or both negative)
                # For the first iteration, prev_diff might be 0, so we allow any sign
                if depth == 0 or (prev_diff * diff > 0) or prev_diff == 0:
                    new_angle = self.calculate_angle(self.nodes[current_node], self.nodes[next_node])
                    
                    # Add edge to current answer
                    self.cur_ans.append((current_node, next_node))
                    
                    self.dfs(next_node, current_node, new_angle, diff, depth + 1)
                    
                    # Backtrack - remove edge from current answer
                    self.cur_ans.pop()
        
        # Backtrack
        self.visited.remove(current_node)
    
    def find_circles(self, max_node):
        """Find all circles in the graph"""
        self.fans.clear()
        
        for node in range(1, max_node + 1):
            if node not in self.adj:
                continue
                
            for adj_node in self.adj[node]:
                self.visited.clear()
                self.cur_ans.clear()
                
                # Set the global start node
                self.start_node = node
                
                # Start DFS from adj_node with node as previous
                initial_angle = self.calculate_angle(self.nodes[node], self.nodes[adj_node])
                
                # Add the first edge
                self.cur_ans.append((node, adj_node))
                self.visited.add(node)
                
                # Start with depth 0 and prev_diff 0
                self.dfs(adj_node, node, initial_angle, 0.0, 0)
    
    def get_all_circles(self):
        """Return all detected circles"""
        return self.fans
    
    def print_circles(self):
        """Print all detected circles"""
        print(f"Found {len(self.fans)} circles:")
        for i, circle in enumerate(self.fans):
            print(f"Circle {i + 1}: {circle}")
    
    def filter_circles_sequential(self, limit=10):
        """
        Filter circles sequentially - only keep a circle if adding it doesn't 
        cause any node to exceed the limit.
        
        This is the exact algorithm you specified:
        - Process circles one by one
        - For each circle, increment node counts
        - If any node exceeds limit, reject circle and decrement counts
        - If all nodes are within limit, keep the circle
        
        Args:
            limit (int): Maximum number of times a node can appear in kept circles
            
        Returns:
            list: Filtered list of circles
        """
        mmp = {}  # map<int,int> equivalent
        res = []  # vector<vector<pair<int,int>>> equivalent
        
        for circle in self.fans:  # for(auto& C : fans)
            is_pos = True  # bool isPos = true
            
            # First pass: increment counts and check if any exceed limit
            for edge in circle:  # for(auto& [u, v]: C)
                u, v = edge
                mmp[u] = mmp.get(u, 0) + 1  # mmp[u]++
                mmp[v] = mmp.get(v, 0) + 1  # mmp[v]++
                
                if mmp[u] > limit or mmp[v] > limit:  # if(mmp[u] > limit || mmp[v] > limit)
                    is_pos = False
            
            if is_pos:  # if(isPos)
                res.append(circle)  # res.push_back(C)
            else:
                # Decrement counts since we're rejecting this circle
                for edge in circle:  # for(auto& [u, v]: C)
                    u, v = edge
                    mmp[u] -= 1  # mmp[u]--
                    mmp[v] -= 1  # mmp[v]--
        
        return res
    
    def apply_sequential_filtering(self, limit=10):
        """
        Apply sequential filtering and update fans in-place
        
        Args:
            limit (int): Maximum number of times a node can appear in kept circles
        """
        original_count = len(self.fans)
        self.fans = self.filter_circles_sequential(limit)
        filtered_count = len(self.fans)
        
        print(f"Sequential filtering complete:")
        print(f"- Original circles: {original_count}")
        print(f"- Filtered circles: {filtered_count}")
        print(f"- Removed circles: {original_count - filtered_count}")
        print(f"- Reduction: {((original_count - filtered_count) / original_count * 100):.1f}%" if original_count > 0 else "0%")
        
        return filtered_count

def load_csv_data(nodes_file, edges_file):
    """Load nodes and edges from CSV files - handles different separators"""
    print(f"Loading entire dataset from CSV files...")
    
    # Try to detect separator and load nodes
    try:
        # First try comma separator
        nodes_df = pd.read_csv(nodes_file, sep=',')
        print(f"Loaded nodes with comma separator")
    except:
        try:
            # Try tab separator
            nodes_df = pd.read_csv(nodes_file, sep='\t')
            print(f"Loaded nodes with tab separator")
        except:
            # Try space separator
            nodes_df = pd.read_csv(nodes_file, sep=' ')
            print(f"Loaded nodes with space separator")
    
    print(f"Loaded {len(nodes_df)} nodes")
    print("Nodes columns:", nodes_df.columns.tolist())
    print("First few rows of nodes:")
    print(nodes_df.head())
    
    # Try to detect separator and load edges
    try:
        # First try comma separator
        edges_df = pd.read_csv(edges_file, sep=',')
        print(f"Loaded edges with comma separator")
    except:
        try:
            # Try tab separator
            edges_df = pd.read_csv(edges_file, sep='\t')
            print(f"Loaded edges with tab separator")
        except:
            # Try space separator
            edges_df = pd.read_csv(edges_file, sep=' ')
            print(f"Loaded edges with space separator")
    
    print(f"Loaded {len(edges_df)} edges")
    print("Edges columns:", edges_df.columns.tolist())
    print("First few rows of edges:")
    print(edges_df.head())
    
    # Create nodes dictionary - handle column names with spaces
    nodes = {}
    node_col = nodes_df.columns[0]  # First column (Node)
    x_col = nodes_df.columns[1]     # Second column (X coord)
    y_col = nodes_df.columns[2]     # Third column (Y coord)
    
    print(f"Using columns: Node='{node_col}', X='{x_col}', Y='{y_col}'")
    
    for _, row in nodes_df.iterrows():
        try:
            node_id = int(row[node_col])
            x_coord = float(row[x_col])
            y_coord = float(row[y_col])
            nodes[node_id] = Node(node_id, x_coord, y_coord)
        except (ValueError, TypeError) as e:
            print(f"Skipping invalid node row: {row.to_dict()}, Error: {e}")
            continue
    
    print(f"Successfully created {len(nodes)} node objects")
    
    # Create adjacency list - handle column names
    adj_list = defaultdict(list)
    node1_col = edges_df.columns[0]  # First column (Node1)
    node2_col = edges_df.columns[1]  # Second column (Node2)
    
    print(f"Using edge columns: Node1='{node1_col}', Node2='{node2_col}'")
    
    edges_added = 0
    for _, row in edges_df.iterrows():
        try:
            node1 = int(row[node1_col])
            node2 = int(row[node2_col])
            
            # Only add edges if both nodes exist in nodes dictionary
            if node1 in nodes and node2 in nodes:
                adj_list[node1].append(node2)
                adj_list[node2].append(node1)  # Undirected graph
                edges_added += 1
            else:
                if node1 not in nodes:
                    print(f"Warning: Node {node1} in edge but not in nodes")
                if node2 not in nodes:
                    print(f"Warning: Node {node2} in edge but not in nodes")
        except (ValueError, TypeError) as e:
            print(f"Skipping invalid edge row: {row.to_dict()}, Error: {e}")
            continue
    
    print(f"Successfully added {edges_added} edges")
    
    return nodes, dict(adj_list)

def extract_unique_nodes_from_circles(circles):
    """Extract all unique nodes that are part of any circle"""
    unique_nodes = set()
    
    for circle in circles:
        for edge in circle:
            unique_nodes.add(edge[0])  # First node in edge
            unique_nodes.add(edge[1])  # Second node in edge
    
    return sorted(list(unique_nodes))

def save_circle_nodes_to_csv(nodes_in_circles, output_file="circle_nodes_output.csv"):
    """Save nodes that are part of circles to CSV"""
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Node'])  # Header
        for node in nodes_in_circles:
            writer.writerow([node])
    
    print(f"Saved {len(nodes_in_circles)} nodes to {output_file}")

def main():
    try:
        # Load data from CSV files (entire dataset)
        nodes, adj_list = load_csv_data("nodes_table.csv", "edges_table.csv")
        
        print(f"\nSuccessfully loaded {len(nodes)} nodes and adjacency list with {len(adj_list)} entries")
        
        # Print some sample data for verification
        sample_nodes = list(nodes.keys())[:5]
        print(f"Sample nodes: {sample_nodes}")
        for node_id in sample_nodes:
            node = nodes[node_id]
            print(f"  Node {node_id}: ({node.x}, {node.y})")
        
        # Find the maximum node ID
        max_node = max(nodes.keys()) if nodes else 0
        print(f"Maximum node ID: {max_node}")
        
        # Print adjacency list sample
        sample_adj = list(adj_list.keys())[:5]
        print(f"Sample adjacency entries:")
        for node_id in sample_adj:
            print(f"  Node {node_id}: connected to {adj_list[node_id]}")
        
        # Create circle detector
        detector = CircleDetector(nodes, adj_list)
        
        # Find circles
        print("\nStarting circle detection with enhanced angle logic and depth limiting...")
        print("Using angle difference range: [0.000001, 50] degrees")
        print("Maximum recursion depth: 10000")
        print("Only accepting cycles that return to start node")
        detector.find_circles(max_node)
        
        # Get initial results
        circles = detector.get_all_circles()
        print(f"\nInitial circles found: {len(circles)}")
        
        # ===== APPLY SEQUENTIAL FILTERING =====
        print("\n" + "="*60)
        print("APPLYING SEQUENTIAL CIRCLE FILTERING")
        print("="*60)
        
        # Apply the sequential filtering algorithm you specified
        limit = 10  # You can change this limit
        print(f"Using limit: {limit} (max times a node can appear in kept circles)")
        
        filtered_circles = detector.filter_circles_sequential(limit)
        
        # Alternatively, apply in-place:
        # detector.apply_sequential_filtering(limit)
        # filtered_circles = detector.get_all_circles()
        
        # Extract unique nodes from filtered circles
        nodes_in_filtered_circles = extract_unique_nodes_from_circles(filtered_circles)
        nodes_in_all_circles = extract_unique_nodes_from_circles(circles)
        
        print(f"\nNode Analysis:")
        print(f"- Nodes in original circles: {len(nodes_in_all_circles)}")
        print(f"- Nodes in filtered circles: {len(nodes_in_filtered_circles)}")
        
        # Save results
        save_circle_nodes_to_csv(nodes_in_all_circles, "all_circle_nodes_output.csv")
        save_circle_nodes_to_csv(nodes_in_filtered_circles, "filtered_circle_nodes_output.csv")
        
        # Print final summary
        print(f"\n" + "="*60)
        print("FINAL SUMMARY")
        print("="*60)
        print(f"- Total nodes loaded: {len(nodes)}")
        print(f"- Total edges loaded: {sum(len(adj) for adj in adj_list.values()) // 2}")
        print(f"- Original circles found: {len(circles)}")
        print(f"- Filtered circles (limit={limit}): {len(filtered_circles)}")
        print(f"- Circles removed: {len(circles) - len(filtered_circles)}")
        print(f"- Reduction percentage: {((len(circles) - len(filtered_circles)) / len(circles) * 100):.1f}%" if circles else "0%")
        print(f"- Files saved:")
        print(f"  * all_circle_nodes_output.csv")
        print(f"  * filtered_circle_nodes_output.csv")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find CSV file - {e}")
        print("Please make sure 'nodes_table.csv' and 'edges_table.csv' are in the same directory")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()