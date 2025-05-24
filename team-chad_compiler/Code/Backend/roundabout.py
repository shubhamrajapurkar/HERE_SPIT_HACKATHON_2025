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
    
    def calculate_circle_indegree(self, circle):
        """
        Calculate indegree for a circle C1 = [(u, v), ...]
        
        For each edge (u, v) in the circle:
        - For each adjacent node of u in the graph
        - If the edge [adj, u] is NOT in the circle, increment indegree
        
        Args:
            circle: List of edges [(u, v), ...]
            
        Returns:
            int: Indegree of the circle
        """
        indegree = 0
        
        # Convert circle to a set of edges for faster lookup
        # Store both (u, v) and (v, u) since the graph is undirected
        circle_edges = set()
        for u, v in circle:
            circle_edges.add((u, v))
            circle_edges.add((v, u))
        
        # For each edge in the circle
        for u, v in circle:
            # Check all adjacent nodes of u
            if u in self.adj:
                for adj in self.adj[u]:
                    # If the edge [adj, u] is NOT in the circle
                    if (adj, u) not in circle_edges:
                        indegree += 1
        
        return indegree
    
    def filter_circles_sequential_with_indegree(self, node_limit=30, indegree_threshold=4):
        """
        Filter circles sequentially with both node count limit and indegree threshold.
        
        Process circles one by one:
        1. Check if circle's indegree >= indegree_threshold
        2. If yes, increment node counts and check node_limit
        3. If any node exceeds node_limit, reject circle and decrement counts
        4. If all nodes are within limit, keep the circle
        
        Args:
            node_limit (int): Maximum number of times a node can appear in kept circles
            indegree_threshold (int): Minimum indegree required for a circle to be kept
            
        Returns:
            tuple: (filtered_circles, indegree_stats)
        """
        mmp = {}  # map<int,int> for node counts
        res = []  # filtered circles
        indegree_stats = {
            'total_circles': len(self.fans),
            'passed_indegree': 0,
            'failed_indegree': 0,
            'passed_node_limit': 0,
            'failed_node_limit': 0,
            'indegree_values': []
        }
        
        print(f"Starting sequential filtering with indegree check...")
        print(f"- Node limit: {node_limit}")
        print(f"- Indegree threshold: {indegree_threshold}")
        print(f"- Total circles to process: {len(self.fans)}")
        
        for i, circle in enumerate(self.fans):
            if (i + 1) % 1000 == 0:  # Progress indicator
                print(f"  Processed {i + 1}/{len(self.fans)} circles...")
            
            # Step 1: Calculate indegree for this circle
            circle_indegree = self.calculate_circle_indegree(circle)
            indegree_stats['indegree_values'].append(circle_indegree)
            
            # Step 2: Check indegree threshold
            if circle_indegree < indegree_threshold:
                indegree_stats['failed_indegree'] += 1
                continue  # Skip this circle
            
            indegree_stats['passed_indegree'] += 1
            
            # Step 3: Check node limit (original sequential filtering logic)
            is_pos = True  # bool isPos = true
            
            # First pass: increment counts and check if any exceed limit
            for edge in circle:  # for(auto& [u, v]: C)
                u, v = edge
                mmp[u] = mmp.get(u, 0) + 1  # mmp[u]++
                mmp[v] = mmp.get(v, 0) + 1  # mmp[v]++
                
                if mmp[u] > node_limit or mmp[v] > node_limit:  # if(mmp[u] > limit || mmp[v] > limit)
                    is_pos = False
            
            if is_pos:  # if(isPos)
                res.append(circle)  # res.push_back(C)
                indegree_stats['passed_node_limit'] += 1
            else:
                # Decrement counts since we're rejecting this circle
                for edge in circle:  # for(auto& [u, v]: C)
                    u, v = edge
                    mmp[u] -= 1  # mmp[u]--
                    mmp[v] -= 1  # mmp[v]--
                indegree_stats['failed_node_limit'] += 1
        
        return res, indegree_stats
    
    def apply_sequential_filtering_with_indegree(self, node_limit=10, indegree_threshold=10):
        """
        Apply sequential filtering with indegree check and update fans in-place
        
        Args:
            node_limit (int): Maximum number of times a node can appear in kept circles
            indegree_threshold (int): Minimum indegree required for a circle to be kept
        """
        original_count = len(self.fans)
        
        # Apply the filtering
        filtered_circles, stats = self.filter_circles_sequential_with_indegree(
            node_limit, indegree_threshold
        )
        
        # Update fans in-place
        self.fans = filtered_circles
        filtered_count = len(self.fans)
        
        # Print detailed statistics
        print(f"\nSequential filtering with indegree check complete:")
        print(f"- Original circles: {original_count}")
        print(f"- Failed indegree check (< {indegree_threshold}): {stats['failed_indegree']}")
        print(f"- Passed indegree check: {stats['passed_indegree']}")
        print(f"- Failed node limit check: {stats['failed_node_limit']}")
        print(f"- Final filtered circles: {filtered_count}")
        print(f"- Total removed: {original_count - filtered_count}")
        print(f"- Overall reduction: {((original_count - filtered_count) / original_count * 100):.1f}%" if original_count > 0 else "0%")
        
        if stats['indegree_values']:
            indegree_values = stats['indegree_values']
            print(f"\nIndegree statistics:")
            print(f"- Min indegree: {min(indegree_values)}")
            print(f"- Max indegree: {max(indegree_values)}")
            print(f"- Average indegree: {sum(indegree_values) / len(indegree_values):.2f}")
        
        return filtered_count, stats

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
        
        # ===== APPLY SEQUENTIAL FILTERING WITH INDEGREE CHECK =====
        print("\n" + "="*60)
        print("APPLYING SEQUENTIAL CIRCLE FILTERING WITH INDEGREE CHECK")
        print("="*60)
        
        # Apply the enhanced sequential filtering algorithm
        node_limit = 10        # Max times a node can appear in kept circles
        indegree_threshold = 10  # Min indegree required for a circle
        
        print(f"Using node limit: {node_limit}")
        print(f"Using indegree threshold: {indegree_threshold}")
        
        # Apply filtering and get statistics
        final_count, stats = detector.apply_sequential_filtering_with_indegree(
            node_limit, indegree_threshold
        )
        
        filtered_circles = detector.get_all_circles()
        
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
        print(f"- Circles passed indegree check: {stats['passed_indegree']}")
        print(f"- Circles failed indegree check: {stats['failed_indegree']}")
        print(f"- Final filtered circles: {len(filtered_circles)}")
        print(f"- Total circles removed: {len(circles) - len(filtered_circles)}")
        print(f"- Overall reduction percentage: {((len(circles) - len(filtered_circles)) / len(circles) * 100):.1f}%" if circles else "0%")
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