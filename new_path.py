import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import time
from math import pi, cos, sin

class Node:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.parent = None

class RRT3D:
    def __init__(self, start, goal, building_list, circle_zones, x_range, y_range, z_range, 
                 step_size=1.0, max_iter=1000, space_factor=0.5, cruise_altitude=5.0,
                 takeoff_distance=2.0, landing_distance=2.0, vertical_step=0.5,
                 consider_overflight=True, building_clearance=1.0):
        self.start = Node(*start)
        self.goal = Node(*goal)
        self.building_list = building_list  
        self.circle_zones = circle_zones
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.step_size = step_size
        self.max_iter = max_iter
        self.space_factor = space_factor
        self.cruise_altitude = cruise_altitude
        self.takeoff_distance = takeoff_distance
        self.landing_distance = landing_distance
        self.vertical_step = vertical_step
        self.consider_overflight = consider_overflight
        self.building_clearance = building_clearance
        self.nodes = [self.start]
        
        # Calculate the maximum building height for overflight decisions
        self.max_building_height = max([b[2] + b[5] for b in building_list]) if building_list else 0
        
        # Check if start or goal is inside a no-fly zone
        if self.is_in_circle_zone(start[0], start[1]) or self.is_in_circle_zone(goal[0], goal[1]):
            print("WARNING: Start or goal position is inside a circular no-fly zone!")

    def is_in_circle_zone(self, x, y):
        """Check if a point (x,y) is inside any circular no-fly zone."""
        for cx, cy, radius in self.circle_zones:
            distance = np.sqrt((x - cx)**2 + (y - cy)**2)
            if distance < radius + self.space_factor:
                return True
        return False

    def get_random_point(self):
        """Generate random points without cruise altitude bias"""
        # Add goal bias - 20% chance to return the goal position
        if random.random() < 0.2:
            return (self.goal.x, self.goal.y, self.goal.z)
        
        # No cruise altitude bias - explore all altitudes
        z = random.uniform(*self.z_range)
        
        # Generate random points outside circular no-fly zones
        max_attempts = 20
        for _ in range(max_attempts):
            x = random.uniform(*self.x_range)
            y = random.uniform(*self.y_range)
            if not self.is_in_circle_zone(x, y):
                return (x, y, z)
        
        # If we can't find a point outside no-fly zones, expand from existing nodes
        rand_node = random.choice(self.nodes)
        rand_dir = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)])
        rand_dir = rand_dir / np.linalg.norm(rand_dir) * self.step_size
        
        return (rand_node.x + rand_dir[0], rand_node.y + rand_dir[1], rand_node.z + rand_dir[2])

    def get_nearest_node(self, point):
        return min(self.nodes, key=lambda node: np.linalg.norm([node.x - point[0], node.y - point[1], node.z - point[2]]))

    def is_collision_free(self, p1, p2):
        """ Checks if the line segment between p1 and p2 collides with any building or enters a circular no-fly zone."""
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        # Check if either endpoint is inside a no-fly zone
        if self.is_in_circle_zone(x1, y1) or self.is_in_circle_zone(x2, y2):
            return False

        # Generate multiple points along the line segment for collision checking
        points = 10
        for i in range(points + 1):
            t = i / points
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            z = z1 + t * (z2 - z1)
            
            # Check if point is within environment bounds
            if (x < self.x_range[0] or x > self.x_range[1] or 
                y < self.y_range[0] or y > self.y_range[1] or
                z < self.z_range[0] or z > self.z_range[1]):
                return False
            
            # Check if this point is too close to any building
            for (bx, by, bz, w, d, h) in self.building_list:
                # Check if point is inside building with clearance
                if (bx - self.space_factor <= x <= bx + w + self.space_factor and
                    by - self.space_factor <= y <= by + d + self.space_factor and
                    bz - self.space_factor <= z <= bz + h + self.space_factor):
                    
                    # Exception for overflight if enabled
                    if self.consider_overflight and z > bz + h + self.building_clearance:
                        continue
                        
                    return False
            
            # Check if this point is inside any circular no-fly zone
            if self.is_in_circle_zone(x, y):
                return False

        return True
    
    def is_point_over_building(self, x, y):
        """Check if a point (x,y) is directly over any building."""
        for (bx, by, bz, w, d, h) in self.building_list:
            if bx <= x <= bx + w and by <= y <= by + d:
                return True, bz + h
        return False, 0

    def steer(self, nearest, rnd_point):
        """Steer from nearest node toward random point"""
        # Calculate direction vector
        direction = np.array([rnd_point[0] - nearest.x, rnd_point[1] - nearest.y, rnd_point[2] - nearest.z])
        distance = np.linalg.norm(direction)
        
        # Limit step size
        if distance > self.step_size:
            direction = direction / distance * self.step_size
        
        new_point = (nearest.x + direction[0], nearest.y + direction[1], nearest.z + direction[2])
        return new_point

    def build_rrt(self):
        """Build the RRT with improved termination condition checking"""
        start_time = time.time()
        goal_reached = False
        min_goal_dist = float('inf')
        stagnation_counter = 0
        
        for i in range(self.max_iter):
            if i % 100 == 0 and i > 0:
                elapsed = time.time() - start_time
                print(f"RRT progress: {i}/{self.max_iter} iterations ({elapsed:.2f} seconds)")
                
            rnd_point = self.get_random_point()
            nearest = self.get_nearest_node(rnd_point)
            new_point = self.steer(nearest, rnd_point)
            
            if self.is_collision_free((nearest.x, nearest.y, nearest.z), new_point):
                new_node = Node(*new_point)
                new_node.parent = nearest
                self.nodes.append(new_node)
                
                # Track minimum distance to goal for stagnation detection
                goal_dist = np.linalg.norm([new_node.x - self.goal.x, new_node.y - self.goal.y, new_node.z - self.goal.z])
                min_goal_dist = min(min_goal_dist, goal_dist)
                
                # Check for goal connection
                goal_connection_distance = self.step_size * 3
                if goal_dist < goal_connection_distance:
                    if self.is_collision_free((new_node.x, new_node.y, new_node.z), 
                                             (self.goal.x, self.goal.y, self.goal.z)):
                        self.goal.parent = new_node
                        self.nodes.append(self.goal)
                        goal_reached = True
                        print(f"Path found in {i} iterations ({time.time() - start_time:.2f} seconds)")
                        return self.get_path()
                
                # Stagnation detection
                if i % 100 == 0 and i > 0:
                    if goal_dist == min_goal_dist:
                        stagnation_counter += 1
                    else:
                        stagnation_counter = 0
                    
                    if stagnation_counter >= 3:
                        closest_node = self.get_nearest_node((self.goal.x, self.goal.y, self.goal.z))
                        if self.is_collision_free((closest_node.x, closest_node.y, closest_node.z),
                                                 (self.goal.x, self.goal.y, self.goal.z)):
                            self.goal.parent = closest_node
                            self.nodes.append(self.goal)
                            goal_reached = True
                            print(f"Path found after stagnation detection in {i} iterations")
                            return self.get_path()

        # Final attempt to connect to goal
        if not goal_reached and len(self.nodes) > 1:
            nodes_sorted = sorted(self.nodes, 
                                 key=lambda node: np.linalg.norm([node.x - self.goal.x, 
                                                                 node.y - self.goal.y, 
                                                                 node.z - self.goal.z]))
            
            for node in nodes_sorted[:min(10, len(nodes_sorted))]:
                if self.is_collision_free((node.x, node.y, node.z), 
                                         (self.goal.x, self.goal.y, self.goal.z)):
                    self.goal.parent = node
                    self.nodes.append(self.goal)
                    print(f"Path found in final connection attempt")
                    return self.get_path()

        print(f"No path found after {self.max_iter} iterations ({time.time() - start_time:.2f} seconds)")
        return None

    def get_path(self):
        """ Extracts the path from goal to start. """
        path = []
        node = self.goal
        while node is not None:
            path.append((node.x, node.y, node.z))
            node = node.parent
        return path[::-1]  

    def smooth_path(self, path):
        """ Shortcuts the path by removing unnecessary waypoints while avoiding obstacles. """
        print("Smoothing path...")
        if not path or len(path) < 3:
            return path

        smoothed_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            for j in range(len(path) - 1, i, -1):
                if self.is_collision_free(path[i], path[j]):
                    smoothed_path.append(path[j])
                    i = j
                    break
            else:
                i += 1
                if i < len(path):
                    smoothed_path.append(path[i])

        print(f"Path smoothed: {len(path)} points reduced to {len(smoothed_path)} points")
        return smoothed_path

    def calculate_path_length(self, path):
        """Calculate the total distance of a path."""
        if not path or len(path) < 2:
            return float('inf')
            
        length = 0
        for i in range(len(path) - 1):
            p1 = np.array(path[i])
            p2 = np.array(path[i+1])
            length += np.linalg.norm(p2 - p1)
            
        return length

    def optimize_path_altitude(self, path):
        """Optimize path altitude only when necessary for building clearance"""
        print("Optimizing path altitude...")
        if not path or len(path) < 2:
            return path
            
        optimized_path = [path[0]]
        
        for i in range(1, len(path)):
            x, y, z = path[i]
            prev_x, prev_y, prev_z = optimized_path[-1]
            
            # Check if we're over a building and need to adjust altitude
            over_building, building_height = self.is_point_over_building(x, y)
            
            if over_building:
                # Calculate required clearance height
                required_height = building_height + self.building_clearance
                current_height = z
                
                # Only adjust if we're below required height
                if current_height < required_height:
                    # Create intermediate point at required height
                    intermediate_point = (x, y, required_height)
                    if self.is_collision_free(optimized_path[-1], intermediate_point):
                        optimized_path.append(intermediate_point)
                    else:
                        # If direct ascent not possible, find safe path
                        optimized_path.append((x, y, max(z, required_height)))
                else:
                    optimized_path.append((x, y, z))
            else:
                # Not over building, keep current altitude
                optimized_path.append((x, y, z))
        
        print(f"Path altitude optimized: {len(path)} -> {len(optimized_path)} points")
        return optimized_path

    def find_optimal_path(self):
        """Find the optimal path by comparing distance of going around vs. over buildings."""
        print("Finding optimal path...")
        
        # First try with standard approach (avoiding buildings horizontally)
        self.consider_overflight = False
        around_path = self.build_rrt()
        
        if around_path:
            smoothed_around = self.smooth_path(around_path)
            optimized_around = self.optimize_path_altitude(smoothed_around)
            around_distance = self.calculate_path_length(optimized_around)
            print(f"Path avoiding buildings: {around_distance:.2f} units")
        else:
            print("Could not find path avoiding buildings")
            optimized_around = None
            around_distance = float('inf')
        
        # Reset RRT and try with building overflight
        self.nodes = [self.start]
        self.consider_overflight = True
        over_path = self.build_rrt()
        
        if over_path:
            smoothed_over = self.smooth_path(over_path)
            optimized_over = self.optimize_path_altitude(smoothed_over)
            over_distance = self.calculate_path_length(optimized_over)
            print(f"Path with building overflight: {over_distance:.2f} units")
        else:
            print("Could not find path with building overflight")
            optimized_over = None
            over_distance = float('inf')
        
        # Compare and return the optimal path
        if around_distance == float('inf') and over_distance == float('inf'):
            print("FAILED: Neither approach found a valid path")
            return None, None
        
        if over_distance < around_distance:
            print(f"Selected path OVER buildings (shorter by {around_distance - over_distance:.2f} units)")
            return optimized_over, optimized_around
        else:
            print(f"Selected path AROUND buildings (shorter by {over_distance - around_distance:.2f} units)")
            return optimized_around, optimized_over

    def plot(self, optimal_path, alternative_path=None, title="3D Drone Flight Path"):
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot buildings (cuboids)
        for (bx, by, bz, w, d, h) in self.building_list:
            color = "red" if h > self.cruise_altitude else "gray"
            alpha = 0.9 if h > self.cruise_altitude else 0.7
            ax.bar3d(bx, by, bz, w, d, h, color=color, alpha=alpha, shade=True)
            
            # Label tall buildings
            if h > self.cruise_altitude:
                ax.text(bx + w/2, by + d/2, bz + h + 2, f"Tall: {h}m", 
                       color='red', ha='center', fontsize=8)

        # Plot circular no-fly zones
        theta = np.linspace(0, 2*np.pi, 100)
        for cx, cy, radius in self.circle_zones:
            # Create circle at z=0
            x = cx + radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            z = np.zeros_like(x)
            ax.plot(x, y, z, 'r-', alpha=0.7, linewidth=2)
            
            # Create vertical lines to show no-fly cylinders
            for zh in np.linspace(0, self.z_range[1], 6):
                ax.plot(x, y, np.ones_like(x) * zh, 'r-', alpha=0.2, linewidth=1)
            
            # Add text label
            ax.text(cx, cy, 0, f"No-Fly Zone\nR={radius}", color='red', ha='center')

        # Plot the alternative path if provided
        if alternative_path:
            alt_x, alt_y, alt_z = zip(*alternative_path)
            ax.plot(alt_x, alt_y, alt_z, "b--", linewidth=1, alpha=0.5, label="Alternative Path")

        # Plot the optimal path
        if optimal_path:
            path_x, path_y, path_z = zip(*optimal_path)
            ax.plot(path_x, path_y, path_z, "g", linewidth=3, label="Optimal Flight Path")
            
            # Mark overflights - points where the path directly crosses over buildings
            overflight_points = []
            for i, (x, y, z) in enumerate(optimal_path):
                over_building, height = self.is_point_over_building(x, y)
                if over_building and z > height:
                    overflight_points.append((x, y, z))
            
            if overflight_points:
                of_x, of_y, of_z = zip(*overflight_points)
                ax.scatter(of_x, of_y, of_z, color="purple", marker="^", s=80, label="Building Overflight")

        # Start and goal
        ax.scatter(self.start.x, self.start.y, self.start.z, color="green", marker="o", s=100, label="Start")
        ax.scatter(self.goal.x, self.goal.y, self.goal.z, color="red", marker="x", s=100, label="Goal")

        # Add cruise altitude reference plane
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 2), np.linspace(y_min, y_max, 2))
        zz = np.ones(xx.shape) * self.cruise_altitude
        ax.plot_surface(xx, yy, zz, alpha=0.1, color='cyan')
        
        # Add text annotation for cruise altitude
        ax.text(x_min, y_min, self.cruise_altitude, f"Cruise altitude: {self.cruise_altitude}", color='blue')

        ax.set_xlim(self.x_range)
        ax.set_ylim(self.y_range)
        ax.set_zlim(self.z_range)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        
        # Calculate path lengths if available
        if optimal_path:
            optimal_length = self.calculate_path_length(optimal_path)
            title = f"{title} - Length: {optimal_length:.2f} units"
            if alternative_path:
                alt_length = self.calculate_path_length(alternative_path)
                title += f" (Alternative: {alt_length:.2f} units)"
            ax.set_title(title, fontsize=14)
        else:
            ax.set_title(title)
            
        ax.legend()
        plt.tight_layout()
        plt.show()

# Function to create dense urban environment with tall buildings
def create_dense_urban_environment():
    """Create a dense urban environment with buildings taller than cruise altitude"""
    buildings = []
    circle_zones = []
    
    # Create a dense grid of buildings in the center
    grid_size = 5
    spacing = 15
    base_x, base_y = -40, -40
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = base_x + i * spacing
            y = base_y + j * spacing
            
            # Variable building heights - many taller than typical cruise altitude
            height = random.choice([8, 12, 18, 25, 30, 35, 40])
            width = random.uniform(8, 12)
            depth = random.uniform(8, 12)
            
            buildings.append((x, y, 0, width, depth, height))
    
    # Add some very tall skyscrapers
    skyscrapers = [
        (-60, -60, 0, 15, 15, 50),
        (60, 60, 0, 20, 20, 60),
        (-60, 60, 0, 12, 12, 45),
        (60, -60, 0, 18, 18, 55),
        (0, 0, 0, 25, 25, 70)  # Central super-tall building
    ]
    buildings.extend(skyscrapers)
    
    # Add no-fly zones in between buildings
    circle_zones.extend([
        (-20, -20, 12),
        (20, 20, 10),
        (-20, 20, 8),
        (20, -20, 15),
        (0, -30, 10),
        (-30, 0, 8)
    ])
    
    return buildings, circle_zones

# Function to create maze-like environment
def create_maze_environment():
    """Create a maze-like environment with narrow passages"""
    buildings = []
    circle_zones = []
    
    # Create outer walls
    buildings.extend([
        (-80, -80, 0, 160, 5, 25),  # Bottom wall
        (-80, 75, 0, 160, 5, 25),   # Top wall  
        (-80, -80, 0, 5, 160, 25),  # Left wall
        (75, -80, 0, 5, 160, 25),   # Right wall
    ])
    
    # Create maze interior walls (many tall buildings)
    interior_walls = [
        # Vertical walls
        (-60, -60, 0, 5, 40, 30),
        (-30, -20, 0, 5, 50, 35),
        (0, -80, 0, 5, 40, 28),
        (30, -40, 0, 5, 60, 32),
        (60, -60, 0, 5, 40, 38),
        
        # Horizontal walls  
        (-40, -40, 0, 30, 5, 26),
        (-10, 0, 0, 40, 5, 33),
        (20, 40, 0, 50, 5, 29),
        (-50, 20, 0, 20, 5, 31),
        (40, -20, 0, 30, 5, 36),
    ]
    buildings.extend(interior_walls)
    
    # Add no-fly zones in narrow passages
    circle_zones.extend([
        (-45, 0, 6),
        (15, -50, 7),
        (45, 10, 5),
        (-15, 45, 8),
        (0, -15, 6)
    ])
    
    return buildings, circle_zones

# Function to create industrial complex environment
def create_industrial_environment():
    """Create an industrial complex with very tall structures"""
    buildings = []
    circle_zones = []
    
    # Large industrial buildings
    large_structures = [
        (-70, -70, 0, 40, 30, 45),   # Main factory
        (-70, 30, 0, 35, 35, 50),    # Processing plant
        (30, -70, 0, 30, 40, 55),    # Storage facility
        (30, 30, 0, 45, 25, 60),     # Headquarters
    ]
    buildings.extend(large_structures)
    
    # Tall narrow structures (smokestacks, towers)
    tall_structures = [
        (-40, -40, 0, 8, 8, 80),
        (-40, 0, 0, 6, 6, 75),
        (0, -40, 0, 7, 7, 85),
        (0, 0, 0, 5, 5, 90),
        (40, 40, 0, 4, 4, 95),
    ]
    buildings.extend(tall_structures)
    
    # Medium buildings
    for i in range(-2, 3):
        for j in range(-2, 3):
            if i == 0 and j == 0:
                continue
            x = i * 25
            y = j * 25
            height = random.choice([20, 25, 30, 35, 40])
            buildings.append((x-5, y-5, 0, 10, 10, height))
    
    # No-fly zones around critical areas
    circle_zones.extend([
        (-20, -20, 15),
        (-20, 20, 12),
        (20, -20, 10),
        (20, 20, 18),
        (0, 0, 20)  # Central restricted area
    ])
    
    return buildings, circle_zones

# Test scenarios
def create_test_scenario(scenario_num):
    """Create different test scenarios for path planning"""
    scenarios = {
        1: {
            'start': (-90, -90, 0),
            'goal': (90, 90, 0),
            'title': "Dense Urban Environment - Straight Line Challenge",
            'environment': create_dense_urban_environment
        },
        2: {
            'start': (-90, 90, 0),
            'goal': (90, -90, 0),
            'title': "Maze Environment - Complex Navigation",
            'environment': create_maze_environment
        },
        3: {
            'start': (-90, 0, 0),
            'goal': (90, 0, 0),
            'title': "Industrial Complex - Tall Structures",
            'environment': create_industrial_environment
        },
        4: {
            'start': (0, -90, 0),
            'goal': (0, 90, 0),
            'title': "Vertical Challenge - Many Tall Buildings",
            'environment': create_dense_urban_environment
        },
        5: {
            'start': (-90, -90, 0),
            'goal': (0, 0, 0),
            'title': "Center Navigation - Super Tall Central Building",
            'environment': create_dense_urban_environment
        }
    }
    
    return scenarios.get(scenario_num, scenarios[1])

# Main function to run simulations
def run_simulation(scenario_num=1):
    """Run the RRT path planning simulation for a given scenario"""
    print(f"\n{'='*60}")
    print(f"Running Scenario {scenario_num}")
    print(f"{'='*60}")
    
    # Get scenario parameters
    scenario = create_test_scenario(scenario_num)
    
    # Define environment parameters for all quadrants
    x_range = (-100, 100)
    y_range = (-100, 100)
    z_range = (0, 100)  # Increased for very tall buildings
    
    # Create environment
    buildings, no_fly_zones = scenario['environment']()
    
    # Print environment stats
    tall_buildings = sum(1 for b in buildings if b[5] > 25)
    max_height = max(b[5] for b in buildings) if buildings else 0
    print(f"Environment: {len(buildings)} buildings, {tall_buildings} taller than 25m, max height: {max_height}m")
    print(f"No-fly zones: {len(no_fly_zones)}")
    
    # Create RRT3D planner with NO cruise altitude bias
    rrt_planner = RRT3D(
        start=scenario['start'],
        goal=scenario['goal'],
        building_list=buildings,
        circle_zones=no_fly_zones,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        step_size=8.0,  # Larger step for complex environments
        max_iter=2000,  # More iterations for difficult paths
        space_factor=3.0,
        cruise_altitude=25.0,  # Reference only, not enforced
        takeoff_distance=10.0,
        landing_distance=10.0,
        vertical_step=3.0,
        consider_overflight=True,
        building_clearance=5.0  # Increased clearance for safety
    )
    
    # Find optimal path
    optimal_path, alternative_path = rrt_planner.find_optimal_path()
    
    # Plot the result
    if optimal_path:
        rrt_planner.plot(optimal_path, alternative_path, scenario['title'])
        print(f"Flight path planning successful!")
        
        # Calculate and print path statistics
        path_length = rrt_planner.calculate_path_length(optimal_path)
        avg_altitude = sum(point[2] for point in optimal_path) / len(optimal_path)
        max_altitude = max(point[2] for point in optimal_path)
        
        print(f"Path statistics:")
        print(f"  - Length: {path_length:.2f} units")
        print(f"  - Average altitude: {avg_altitude:.2f} m")
        print(f"  - Maximum altitude: {max_altitude:.2f} m")
        print(f"  - Cruise altitude reference: {rrt_planner.cruise_altitude} m")
        
        # Count overflight segments
        overflight_count = 0
        for point in optimal_path:
            over_building, height = rrt_planner.is_point_over_building(point[0], point[1])
            if over_building and point[2] > height:
                overflight_count += 1
                
        print(f"  - Overflight segments: {overflight_count}")
        
    else:
        print(f"Failed to find a valid path.")
        # Plot anyway to show the environment
        rrt_planner.plot(None, None, scenario['title'] + " - NO PATH FOUND")
    
    return optimal_path is not None

# Run specific scenario or all scenarios
if __name__ == "__main__":
    print("3D Drone Path Planning - Challenging Environments")
    print("Features:")
    print("- No cruise altitude bias - explores all altitudes")
    print("- Dense building layouts with variable heights")
    print("- Buildings taller than cruise altitude")
    print("- Multiple no-fly zones")
    print("- Complex urban, maze, and industrial environments")
    
    # Run individual scenario
    run_simulation(1)  # Change number to test different scenarios
    
    # Or run all scenarios
    # success_count = 0
    # for i in range(1, 6):
    #     if run_simulation(i):
    #         success_count += 1
    #     print("\n" + "="*60 + "\n")
    # 
    # print(f"Summary: {success_count}/5 scenarios completed successfully")
