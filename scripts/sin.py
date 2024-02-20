import numpy as np
import matplotlib.pyplot as plt

def generate_sinusoid_2d(start_point, end_point, num_points=100, frequency=1, amplitude=1, phase=0):
    x = np.linspace(start_point[0], end_point[0], num_points)
    y = amplitude * np.sin(2 * np.pi * frequency * x + phase)
    
    # Use the same x values for both points to ensure they lie on the same line
    x_values = np.linspace(start_point[0], end_point[0], num_points)
    
    # Combine x values with sinusoidal y values
    coordinates = np.column_stack((x_values, y))
    
    # Rotate the coordinates based on the angle formed by the line
    angle = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]])
    rotated_coordinates = np.dot(coordinates - start_point, rotation_matrix.T) + start_point
    
    return rotated_coordinates[:, 0], rotated_coordinates[:, 1]

# Define the start and end points in 2D coordinates
start_point = np.array([0, -0])
end_point = np.array([1, 1])

# Generate sinusoid curve in 2D
x_values, y_values = generate_sinusoid_2d(start_point, end_point)

# Plot the sinusoid curve
plt.plot(x_values, y_values, label='Sinusoid Curve')
plt.scatter([start_point[0], end_point[0]], [start_point[1], end_point[1]], color='red', label='Start/End Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Sinusoid Curve between Two 2D Points')
plt.legend()
plt.grid(True)
plt.show()