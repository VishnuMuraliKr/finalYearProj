import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
NUM_USERS = 100
STEP_SIZE_DRONE = 0.5
STEP_SIZE_USER = 0.2
GRID_SIZE = 10
NETWORK_RADIUS = 4
DRONE_INIT_POSITION = np.random.rand(2) * GRID_SIZE  # Random initial position within the grid

# Generate random user locations within the grid
users = np.random.rand(NUM_USERS, 2) * GRID_SIZE

# Initialize user directions for random walk
user_directions = np.random.rand(NUM_USERS, 2) - 0.5
user_directions /= np.linalg.norm(user_directions, axis=1)[:, np.newaxis]

# Cost function: Sum of distances from the drone to all users
def cost_function(drone_position, users):
    distances = np.linalg.norm(users - drone_position, axis=1)
    return np.sum(distances)

# Compute the gradient of the cost function (used to determine the direction of movement)
def compute_gradient(drone_position, users):
    distances = np.linalg.norm(users - drone_position, axis=1)
    direction = np.sum((drone_position - users) / distances[:, np.newaxis], axis=0)
    return direction / np.linalg.norm(direction)

# Update the drone's position to move towards the area with maximum users
def update_drone_position(drone_position, users):
    gradient = compute_gradient(drone_position, users)
    new_position = drone_position - STEP_SIZE_DRONE * gradient
    return new_position

# Update users' positions using a random walk with boundary reflection
def update_user_positions(users, user_directions):
    # Randomly change direction slightly
    user_directions += (np.random.rand(NUM_USERS, 2) - 0.5) * 0.1
    user_directions /= np.linalg.norm(user_directions, axis=1)[:, np.newaxis]
    
    # Move users in the direction they're currently facing
    users += STEP_SIZE_USER * user_directions
    
    # Reflect direction if users hit the grid boundaries
    for i in range(NUM_USERS):
        for j in range(2):  # Check both x and y coordinates
            if users[i, j] <= 0 or users[i, j] >= GRID_SIZE:
                user_directions[i, j] *= -1  # Reverse the direction in the offending axis
                users[i, j] = np.clip(users[i, j], 0, GRID_SIZE)  # Keep within bounds
    
    return users, user_directions

# Initialize the figure and axis
fig, ax = plt.subplots()
sc_users = ax.scatter(users[:, 0], users[:, 1], c='blue', label='Users')
sc_drone = ax.scatter(DRONE_INIT_POSITION[0], DRONE_INIT_POSITION[1], c='red', label='Drone')
network_circle = plt.Circle(DRONE_INIT_POSITION, NETWORK_RADIUS, color='green', fill=False, linestyle='--')
ax.add_patch(network_circle)
ax.set_xlim(0, GRID_SIZE)
ax.set_ylim(0, GRID_SIZE)
ax.set_aspect('equal', 'box')  # Ensure the aspect ratio is equal to keep the circle round
ax.legend()

# Update function for the animation
def animate(i):
    global DRONE_INIT_POSITION, users, user_directions
    users, user_directions = update_user_positions(users, user_directions)
    DRONE_INIT_POSITION = update_drone_position(DRONE_INIT_POSITION, users)
    sc_users.set_offsets(users)
    sc_drone.set_offsets(DRONE_INIT_POSITION)
    network_circle.set_center(DRONE_INIT_POSITION)

# Create animation
ani = FuncAnimation(fig, animate, frames=200, interval=100, repeat=False)

plt.show()
