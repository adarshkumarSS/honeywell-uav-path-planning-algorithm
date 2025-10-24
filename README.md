# Path Planning
the below is the output of the python code which achieve optimised path for UAV traversal with RRT algorithm 
![image](https://github.com/user-attachments/assets/d3eafa35-d1aa-4cee-acf6-90c4d12afc4b)
![image](https://github.com/user-attachments/assets/440a9087-085f-4f72-8564-0f68202e0c4c)
![image](https://github.com/user-attachments/assets/f1ecfdc1-faa7-49f1-bf15-dc2db4cf8f15)

🚀 3D Drone Path Planning with RRT Algorithm - A Deep Dive into Urban Navigation 🚀

As drones become increasingly vital for urban applications (delivery, inspection, emergency response), safe and efficient path planning in complex 3D environments is critical. I built a 3D Rapidly-exploring Random Tree (RRT) algorithm to solve this challenge—here’s how it works:

🔍 The Problem
Drones navigating cities must:
✅ Avoid buildings & no-fly zones
✅ Optimize for shortest path
✅ Handle takeoff/landing safely
✅ Decide when to fly over vs. around obstacles

🛠 The Solution: Enhanced RRT Algorithm
My implementation extends classic RRT with:

3D collision detection for buildings (cuboids) + cylindrical no-fly zones

Adaptive altitude control: Cruise altitude optimization with smooth takeoff/landing transitions

Overflight decision-making: Dynamically evaluates if flying over buildings (with clearance) is shorter than navigating around

Path smoothing & stagnation detection for faster convergence

💡 Key Features
1️⃣ Intelligent Sampling: Biases random nodes toward goal and cruise altitude
2️⃣ Safety Buffers: Configurable space_factor ensures minimum distance from obstacles
3️⃣ Multi-Strategy Planning: Compares "overflight" vs. "ground-hugging" paths to pick the optimal one
4️⃣ Visualization: 3D Matplotlib plots show path, buildings, no-fly zones, and key waypoints

📊 Example Output
In a simulated 100x100m urban grid with 4 buildings and 2 no-fly zones, the algorithm:

Found a collision-free path in ~500 iterations

Reduced path length by 22% via smoothing

Automatically chose to fly over a low-rise building instead of detouring

# Try it yourself! 
rrt_planner = RRT3D(
    start=(10, 10, 0),
    goal=(90, 90, 0),
    building_list=buildings,
    circle_zones=no_fly_zones,
    cruise_altitude=15.0,
    space_factor=3.0  # 3m clearance from obstacles
)
optimal_path, _ = rrt_planner.find_optimal_path()

🌟 Why This Matters
This project demonstrates how algorithmic path planning can tackle real-world constraints like:

Regulatory no-fly zones

Dynamic urban landscapes

Energy efficiency (shorter paths = longer battery life)

Next Steps: Integrate real-time obstacle avoidance or machine learning for adaptive altitude selection!
