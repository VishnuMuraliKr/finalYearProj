import numpy as np
from scipy.optimize import minimize
import random

# Define the cost function (sum of weighted distances)
def cost_function(point, points, h):
    # point is [x, y, z] and points is an array of 3D points [[x1, y1, z1], [x2, y2, z2], ...]
    distances = np.linalg.norm(points - point, axis=1)  # Compute distances
    weighted_distances = (distances)**2 * h  # Multiply each distance by the corresponding h value
    return np.sum(weighted_distances)  # Sum of weighted distances

# Define the constraint function for the logarithmic summation
def log_constraint(point, points, h, c):
    distances = np.linalg.norm(points - point, axis=1)  # Euclidean distances
    log_sum = np.sum(np.log2(1 + 1 / (distances * h)))  # Summation of log2(1 + 1 / (distance * h))
    return log_sum - c  # Constraint: log_sum >= c

# Sample points in 3D space (N points)
points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 3, 5], [2, 4, 6]])

points = []
for _ in range(100):
    # Generate random values for x, y, z
    x = random.randint(0, 100)
    y = random.randint(0, 100)
    z = random.randint(0, 10)
    # Append the list [x, y, z] to the matrix
    points.append([x, y, z])

# Create a list h containing 5 random values (either 0.1 or 21)
h = np.random.choice([0.1, 21], size=100)

# Initial guess for the point's position (x0, y0, z0)
initial_guess = np.array([0.0, 0.0, 0.0])

# Set the fixed value for z
z_fixed = 100

# Define the constraint as a dictionary for z < z_fixed
constraint_z = {'type': 'ineq', 'fun': lambda point: z_fixed - point[2]}

# Define the logarithmic summation constraint (log_sum >= c)
c = 1000
constraint_log = {'type': 'ineq', 'fun': log_constraint, 'args': (points, h, c)}

# Optimize the cost function with both constraints (z and log summation)
result = minimize(cost_function, initial_guess, args=(points, h), method='BFGS', constraints=[constraint_z, constraint_log])

# Output the optimized point and the minimized cost
print("Random list h:", h)
print("Optimized point:", result.x)
print("Minimum cost (sum of weighted distances and pathloss product):", result.fun)
