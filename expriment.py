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
        self.circle_zones = circle_zones  # List of (x, y, radius) tuples for circular no-fly zones
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
            if distance < radius + self.space_factor:  # Add space_factor for clearance
                return True
        return False

    def get_random_point(self):
        """Generate random points with improved goal bias and cruise altitude handling"""
        # Add goal bias - 20% chance to return the goal position
        if random.random() < 0.2:
            return (self.goal.x, self.goal.y, self.goal.z)
            
        # Bias random points to be at cruise altitude once reached
        if any(node.z >= self.cruise_altitude for node in self.nodes):
            z = self.cruise_altitude if random.random() < 0.8 else random.uniform(*self.z_range)
        else:
            z = random.uniform(self.start.z, self.cruise_altitude)
        
        # Generate random points outside circular no-fly zones with improved retry logic
        max_attempts = 20
        for _ in range(max_attempts):
            x = random.uniform(*self.x_range)
            y = random.uniform(*self.y_range)
            if not self.is_in_circle_zone(x, y):
                return (x, y, z)
        
        # If we can't find a point outside no-fly zones, focus on expanding from existing nodes
        rand_node = random.choice(self.nodes)
        rand_dir = np.array([random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-0.2, 0.2)])
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
        points = 10  # Number of points to check along the segment
        for i in range(points + 1):
            t = i / points
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            z = z1 + t * (z2 - z1)
            
            # Check if point is within environment bounds
            if (x < self.x_range[0] or x > self.x_range[1] or 
                y < self.y_range[0] or y > self.y_range[1] or
                z < self.z_range[0] or z > self.z_range[1]):
                return False  # Out of bounds
            
            # Check if this point is too close to any building
            for (bx, by, bz, w, d, h) in self.building_list:
                # Check if point is inside building with clearance
                if (bx - self.space_factor <= x <= bx + w + self.space_factor and
                    by - self.space_factor <= y <= by + d + self.space_factor and
                    bz - self.space_factor <= z <= bz + h + self.space_factor):
                    
                    # Exception for overflight if enabled
                    if self.consider_overflight and z > bz + h + self.building_clearance:
                        continue  # Allow flying over buildings with sufficient clearance
                        
                    return False  # Too close to building
            
            # Check if this point is inside any circular no-fly zone
            if self.is_in_circle_zone(x, y):
                # No exception for overflight of no-fly zones
                return False  # Inside no-fly zone

        return True
    
    def is_point_over_building(self, x, y):
        """Check if a point (x,y) is directly over any building."""
        for (bx, by, bz, w, d, h) in self.building_list:
            if bx <= x <= bx + w and by <= y <= by + d:
                return True, bz + h  # Return True and the building height
        return False, 0

    def steer(self, nearest, rnd_point):
        """Steer from nearest node toward random point with improved handling of vertical motion"""
        # Calculate direction vector
        direction = np.array([rnd_point[0] - nearest.x, rnd_point[1] - nearest.y, rnd_point[2] - nearest.z])
        distance = np.linalg.norm(direction)
        
        # Limit step size
        if distance > self.step_size:
            direction = direction / distance * self.step_size
        
        # Prioritize vertical motion when appropriate
        if ((abs(nearest.z - self.start.z) < 2.0 and distance < 3.0) or 
            (abs(nearest.z - self.goal.z) < 2.0 and distance < 3.0) or
            (abs(nearest.z - self.cruise_altitude) < 1.0 and nearest.z < self.cruise_altitude)):
            # Emphasize vertical motion
            direction[2] *= 1.5
            # Renormalize to maintain step size
            direction = direction / np.linalg.norm(direction) * min(self.step_size, distance)
        
        new_point = (nearest.x + direction[0], nearest.y + direction[1], nearest.z + direction[2])
        return new_point

    def build_rrt(self):
        """Build the RRT with improved termination condition checking"""
        start_time = time.time()
        goal_reached = False
        min_goal_dist = float('inf')
        stagnation_counter = 0
        
        for i in range(self.max_iter):
            # Print progress every 100 iterations
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
                
                # Check for goal connection more aggressively
                goal_connection_distance = self.step_size * 3
                if goal_dist < goal_connection_distance:
                    if self.is_collision_free((new_node.x, new_node.y, new_node.z), 
                                             (self.goal.x, self.goal.y, self.goal.z)):
                        self.goal.parent = new_node
                        self.nodes.append(self.goal)
                        goal_reached = True
                        print(f"Path found in {i} iterations ({time.time() - start_time:.2f} seconds)")
                        return self.get_path()
                
                # Stagnation detection - if no improvement in 100 iterations, try direct goal connection
                if i % 100 == 0 and i > 0:
                    if goal_dist == min_goal_dist:
                        stagnation_counter += 1
                    else:
                        stagnation_counter = 0
                    
                    # After 3 stagnation checks (300 iterations), try connecting to goal from closest node
                    if stagnation_counter >= 3:
                        closest_node = self.get_nearest_node((self.goal.x, self.goal.y, self.goal.z))
                        if self.is_collision_free((closest_node.x, closest_node.y, closest_node.z),
                                                 (self.goal.x, self.goal.y, self.goal.z)):
                            self.goal.parent = closest_node
                            self.nodes.append(self.goal)
                            goal_reached = True
                            print(f"Path found after stagnation detection in {i} iterations")
                            return self.get_path()

        # If we've exhausted iterations but have nodes, try connecting any node to goal
        if not goal_reached and len(self.nodes) > 1:
            # Sort nodes by distance to goal
            nodes_sorted = sorted(self.nodes, 
                                 key=lambda node: np.linalg.norm([node.x - self.goal.x, 
                                                                 node.y - self.goal.y, 
                                                                 node.z - self.goal.z]))
            
            # Try connecting to goal from the closest few nodes
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
                if self.is_collision_free(path[i], path[j]):  # Direct shortcut possible
                    smoothed_path.append(path[j])
                    i = j
                    break
            else:
                # No shortcut found
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

    def optimize_for_altitude(self, path):
        """Optimizes the path to maintain cruise altitude except during takeoff and landing."""
        print("Optimizing path for altitude...")
        if not path or len(path) < 2:
            return path
            
        optimized_path = []
        
        # Add the starting point
        optimized_path.append(path[0])
        
        # Generate takeoff path points
        start_x, start_y, start_z = path[0]
        
        # Calculate takeoff phase (vertical first, then transition to cruise)
        takeoff_points = []
        vertical_rise_points = max(1, int(self.cruise_altitude / self.vertical_step))
        
        for i in range(1, vertical_rise_points + 1):
            z = start_z + i * self.vertical_step
            takeoff_points.append((start_x, start_y, min(z, self.cruise_altitude)))
            
        # Add transition phase - gradually move horizontally while at cruise altitude
        if len(path) > 2:
            mid_point = path[2]  # Use a point further along the path for direction
        else:
            mid_point = path[-1]  # If no other points, use the goal
            
        transition_steps = 3
        for i in range(1, transition_steps + 1):
            t = i / transition_steps
            x = start_x + t * (mid_point[0] - start_x)
            y = start_y + t * (mid_point[1] - start_y)
            takeoff_points.append((x, y, self.cruise_altitude))
            
        # Validate takeoff points don't enter no-fly zones
        valid_takeoff_points = []
        prev_point = path[0]
        for point in takeoff_points:
            if not self.is_in_circle_zone(point[0], point[1]) and self.is_collision_free(prev_point, point):
                valid_takeoff_points.append(point)
                prev_point = point
        
        optimized_path.extend(valid_takeoff_points)
        
        # Main cruise path - maintain cruise altitude while considering building overflights and no-fly zones
        cruise_points = []
        for i in range(1, len(path) - 1):
            x, y, _ = path[i]
            
            # Skip points inside no-fly zones
            if self.is_in_circle_zone(x, y):
                continue
                
            over_building, building_height = self.is_point_over_building(x, y)
            
            # Decide whether to go over or around the building
            if over_building and self.consider_overflight:
                # Calculate required clearance height
                clearance_height = building_height + self.building_clearance
                
                if clearance_height > self.cruise_altitude:
                    # Need to go higher to clear this building
                    cruise_points.append((x, y, clearance_height))
                else:
                    # Can maintain cruise altitude
                    cruise_points.append((x, y, self.cruise_altitude))
            else:
                # Not over a building, maintain cruise altitude
                cruise_points.append((x, y, self.cruise_altitude))
        
        # Add intermediate path points only if collision-free
        if cruise_points:
            prev_point = optimized_path[-1]
            for point in cruise_points:
                if self.is_collision_free(prev_point, point):
                    optimized_path.append(point)
                    prev_point = point
        
        # Generate landing path
        goal_x, goal_y, goal_z = path[-1]
        
        # Get the last point of the current path
        last_x, last_y, last_z = optimized_path[-1]
        
        # Approach phase - move toward landing spot while maintaining altitude
        approach_distance = np.linalg.norm([goal_x - last_x, goal_y - last_y])
        if approach_distance > self.landing_distance:
            direction = np.array([goal_x - last_x, goal_y - last_y])
            direction = direction / approach_distance
            approach_x = goal_x - direction[0] * self.landing_distance
            approach_y = goal_y - direction[1] * self.landing_distance
            # Check if approach point is outside no-fly zones
            approach_point = (approach_x, approach_y, last_z)
            if not self.is_in_circle_zone(approach_x, approach_y) and self.is_collision_free(optimized_path[-1], approach_point):
                optimized_path.append(approach_point)
        
        # Descent phase - gradual descent to landing spot
        descent_points = []
        descent_steps = max(1, int(optimized_path[-1][2] / self.vertical_step))
        
        for i in range(1, descent_steps + 1):
            t = i / descent_steps
            x = optimized_path[-1][0] + t * (goal_x - optimized_path[-1][0])
            y = optimized_path[-1][1] + t * (goal_y - optimized_path[-1][1])
            z = optimized_path[-1][2] - t * (optimized_path[-1][2] - goal_z)
            # Only add points that are outside no-fly zones
            descent_point = (x, y, z)
            if not self.is_in_circle_zone(x, y) and (not descent_points or self.is_collision_free(descent_points[-1], descent_point)):
                descent_points.append(descent_point)
            
        optimized_path.extend(descent_points)
        
        # Add the goal point if not already included and connection is collision-free
        if optimized_path[-1] != path[-1] and self.is_collision_free(optimized_path[-1], path[-1]):
            optimized_path.append(path[-1])
        
        print(f"Path altitude-optimized: {len(path)} points transformed to {len(optimized_path)} points")
        return optimized_path

    def find_optimal_path(self):
        """Find the optimal path by comparing distance of going around vs. over buildings."""
        print("Finding optimal path...")
        
        # First try with standard cruise altitude (avoiding buildings horizontally)
        self.consider_overflight = False
        around_path = self.build_rrt()
        
        if around_path:
            smoothed_around = self.smooth_path(around_path)
            optimized_around = self.optimize_for_altitude(smoothed_around)
            around_distance = self.calculate_path_length(optimized_around)
            print(f"Path avoiding buildings horizontally: {around_distance:.2f} units")
        else:
            print("Could not find path avoiding buildings horizontally")
            optimized_around = None
            around_distance = float('inf')
        
        # Reset RRT and try with building overflight
        self.nodes = [self.start]
        self.consider_overflight = True
        over_path = self.build_rrt()
        
        if over_path:
            smoothed_over = self.smooth_path(over_path)
            optimized_over = self.optimize_for_altitude(smoothed_over)
            over_distance = self.calculate_path_length(optimized_over)
            print(f"Path considering building overflight: {over_distance:.2f} units")
        else:
            print("Could not find path considering building overflight")
            optimized_over = None
            over_distance = float('inf')
        
        # Compare and return the optimal path
        if around_distance == float('inf') and over_distance == float('inf'):
            print("FAILED: Neither approach found a valid path")
            # Last resort - attempt a simple direct path
            self.nodes = [self.start]
            self.max_iter = self.max_iter * 2  # Double max iterations
            self.step_size = self.step_size / 2  # Halve step size for finer resolution
            self.consider_overflight = True  # Allow overflight
            last_resort_path = self.build_rrt()
            
            if last_resort_path:
                optimized_last_resort = self.optimize_for_altitude(last_resort_path)
                print("Found a path with emergency settings")
                return optimized_last_resort, None
            return None, None
        
        if over_distance < around_distance:
            print(f"Selected path OVER buildings (shorter by {around_distance - over_distance:.2f} units)")
            return optimized_over, optimized_around
        else:
            print(f"Selected path AROUND buildings (shorter by {over_distance - around_distance:.2f} units)")
            return optimized_around, optimized_over

    def plot(self, optimal_path, alternative_path=None, title="3D Drone Flight Path"):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot buildings (cuboids)
        for (bx, by, bz, w, d, h) in self.building_list:
            ax.bar3d(bx, by, bz, w, d, h, color="gray", alpha=0.8, shade=True)

        # Plot circular no-fly zones
        theta = np.linspace(0, 2*np.pi, 100)
        for cx, cy, radius in self.circle_zones:
            # Create circle at z=0
            x = cx + radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            z = np.zeros_like(x)
            ax.plot(x, y, z, 'r-', alpha=0.7, linewidth=2)
            
            # Create vertical lines to show no-fly cylinders
            for zh in np.linspace(0, self.cruise_altitude * 1.5, 4):
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
            
            # Highlight takeoff and landing segments
            # First 5 points typically represent takeoff
            takeoff_index = min(5, len(optimal_path)//3)
            if takeoff_index > 1:
                takeoff_x, takeoff_y, takeoff_z = zip(*optimal_path[:takeoff_index])
                ax.plot(takeoff_x, takeoff_y, takeoff_z, "r-", linewidth=4, label="Takeoff")
            
            # Last 5 points typically represent landing
            landing_index = max(len(optimal_path) - 5, 2*len(optimal_path)//3)
            if landing_index < len(optimal_path) - 1:
                landing_x, landing_y, landing_z = zip(*optimal_path[landing_index:])
                ax.plot(landing_x, landing_y, landing_z, "y-", linewidth=4, label="Landing")
            
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

        # Add a horizontal plane at cruise altitude
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
            title = f"{title} - Optimal Length: {optimal_length:.2f} units"
            if alternative_path:
                alt_length = self.calculate_path_length(alternative_path)
                title += f" (Alternative: {alt_length:.2f} units)"
            ax.set_title(title)
        else:
            ax.set_title(title)
            
        ax.legend()
        plt.tight_layout()
        plt.show()

# Function to create obstacles in all 8 quadrants
def create_8_quadrant_obstacles():
    """Create obstacles distributed across all 8 quadrants"""
    buildings = []
    circle_zones = []
    
    # Quadrant I (+, +)
    buildings.extend([
        (20, 20, 0, 10, 10, 15),   # Building 1
        (50, 40, 0, 15, 12, 20),   # Building 2
    ])
    circle_zones.append((35, 35, 8))  # Circular zone
    
    # Quadrant II (-, +)
    buildings.extend([
        (-30, 25, 0, 12, 8, 18),   # Building 3
        (-60, 50, 0, 10, 15, 12),  # Building 4
    ])
    circle_zones.append((-40, 40, 10))  # Circular zone
    
    # Quadrant III (-, -)
    buildings.extend([
        (-25, -30, 0, 15, 10, 16),  # Building 5
        (-50, -60, 0, 12, 12, 14),  # Building 6
    ])
    circle_zones.append((-35, -45, 9))  # Circular zone
    
    # Quadrant IV (+, -)
    buildings.extend([
        (30, -25, 0, 10, 15, 17),   # Building 7
        (60, -50, 0, 14, 10, 13),   # Building 8
    ])
    circle_zones.append((45, -35, 7))  # Circular zone
    
    return buildings, circle_zones

# Function to create test scenarios for different quadrants
def create_test_scenario(scenario_num):
    """Create different test scenarios for path planning"""
    scenarios = {
        1: {  # Cross quadrant navigation
            'start': (-80, -80, 0),
            'goal': (80, 80, 0),
            'title': "Cross-Quadrant Navigation (QIII to QI)"
        },
        2: {  # Adjacent quadrant navigation
            'start': (-80, 80, 0),
            'goal': (80, -80, 0),
            'title': "Adjacent Quadrant Navigation (QII to QIV)"
        },
        3: {  # Same quadrant navigation
            'start': (10, 10, 0),
            'goal': (70, 70, 0),
            'title': "Same Quadrant Navigation (QI)"
        },
        4: {  # Negative quadrant navigation
            'start': (-10, -10, 0),
            'goal': (-70, -70, 0),
            'title': "Same Quadrant Navigation (QIII)"
        },
        5: {  # Complex multi-quadrant path
            'start': (-90, 90, 0),
            'goal': (90, -90, 0),
            'title': "Complex Multi-Quadrant Path"
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
    z_range = (0, 30)
    
    # Create obstacles distributed across all quadrants
    buildings, no_fly_zones = create_8_quadrant_obstacles()
    
    # Create RRT3D planner
    rrt_planner = RRT3D(
        start=scenario['start'],
        goal=scenario['goal'],
        building_list=buildings,
        circle_zones=no_fly_zones,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        step_size=5.0,  # Increased for larger space
        max_iter=1500,  # Increased for complex environments
        space_factor=3.0,
        cruise_altitude=25.0,  # Increased for taller buildings
        takeoff_distance=8.0,
        landing_distance=8.0,
        vertical_step=2.0,
        consider_overflight=True,
        building_clearance=3.0
    )
    
    # Find optimal path
    optimal_path, alternative_path = rrt_planner.find_optimal_path()
    
    # Plot the result
    if optimal_path:
        rrt_planner.plot(optimal_path, alternative_path, scenario['title'])
        print(f"Flight path planning successful for {scenario['title']}!")
        
        # Print quadrant information
        start_quadrant = get_quadrant(scenario['start'][0], scenario['start'][1])
        goal_quadrant = get_quadrant(scenario['goal'][0], scenario['goal'][1])
        print(f"Start in {start_quadrant}, Goal in {goal_quadrant}")
        print(f"Path length: {rrt_planner.calculate_path_length(optimal_path):.2f} units")
    else:
        print(f"Failed to find a valid path for {scenario['title']}.")
    
    return optimal_path is not None

def get_quadrant(x, y):
    """Determine which quadrant a point is in"""
    if x >= 0 and y >= 0:
        return "Quadrant I (+, +)"
    elif x < 0 and y >= 0:
        return "Quadrant II (-, +)"
    elif x < 0 and y < 0:
        return "Quadrant III (-, -)"
    else:
        return "Quadrant IV (+, -)"

# Run all scenarios
if __name__ == "__main__":
    print("3D Drone Path Planning in All 8 Quadrants")
    print("This simulation demonstrates path planning across all coordinate quadrants")
    
    # Run individual scenario
    # run_simulation(1)
    
    # Run all scenarios
    success_count = 0
    for i in range(1, 6):
        if run_simulation(i):
            success_count += 1
        print()  # Empty line between scenarios
    
    print(f"\nSummary: {success_count}/5 scenarios completed successfully")
