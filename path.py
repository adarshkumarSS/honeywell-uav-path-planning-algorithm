import numpy as np
import networkx as nx
import json

def generate_mock_data():
    """Creates mock data with waypoints, terrain height, buildings, and prone areas."""
    grid_size = (10, 10)  # 10x10 grid
    waypoints = [(i, j) for i in range(grid_size[0]) for j in range(grid_size[1])]
    
    terrain_height = {wp: np.random.randint(0, 50) for wp in waypoints}  # Random heights (0-50m)
    building_height = {wp: np.random.randint(0, 100) if np.random.rand() > 0.8 else 0 for wp in waypoints}  # 20% chance of buildings (0-100m)
    prone_areas = set([wp for wp in waypoints if np.random.rand() > 0.85])  # 15% of places are prone areas
    
    return waypoints, terrain_height, building_height, prone_areas

def create_graph(waypoints, terrain_height, building_height, prone_areas):
    """Creates a weighted graph for path planning."""
    G = nx.Graph()
    
    for x, y in waypoints:
        if (x, y) in prone_areas:
            continue  # Skip prone areas
        
        G.add_node((x, y), terrain=terrain_height[(x, y)], building=building_height[(x, y)])
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Four directions
            neighbor = (x + dx, y + dy)
            if neighbor in waypoints and neighbor not in prone_areas:
                height_penalty = abs(terrain_height[(x, y)] - terrain_height[neighbor])
                building_penalty = building_height[neighbor]
                weight = 1 + height_penalty + building_penalty
                G.add_edge((x, y), neighbor, weight=weight)
    
    return G

def find_optimal_path(G, start, end):
    """Finds the shortest path considering weights."""
    try:
        path = nx.shortest_path(G, source=start, target=end, weight='weight')
        return path
    except nx.NetworkXNoPath:
        return None

def save_waypoints_arducopter(path, filename="waypoints.waypoints"):
    """Saves waypoints in ArduCopter format."""
    if path:
        with open(filename, "w") as f:
            f.write("QGC WPL 110\n")  # ArduPilot header
            for idx, (x, y) in enumerate(path):
                f.write(f"{idx}\t0\t3\t16\t0\t0\t0\t0\t{x * 0.0001}\t{y * 0.0001}\t50\t1\n")
        print(f"Waypoints saved to {filename}")
    else:
        print("No valid path found.")

# Generate mock data
waypoints, terrain_height, building_height, prone_areas = generate_mock_data()

# Create graph
G = create_graph(waypoints, terrain_height, building_height, prone_areas)

# Define start and end points
start, end = (0, 0), (9, 9)

# Find optimal path
optimal_path = find_optimal_path(G, start, end)

# Save to ArduCopter format
save_waypoints_arducopter(optimal_path)
