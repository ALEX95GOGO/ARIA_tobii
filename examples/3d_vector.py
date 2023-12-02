import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.quiver import Quiver

# Function to rotate a vector around an axis
def rotate_vector(vector, axis, angle):
    axis = axis / np.linalg.norm(axis)  # Normalize axis
    rotation_matrix = np.array([[np.cos(angle) + axis[0]**2 * (1 - np.cos(angle)),
                                axis[0] * axis[1] * (1 - np.cos(angle)) - axis[2] * np.sin(angle),
                                axis[0] * axis[2] * (1 - np.cos(angle)) + axis[1] * np.sin(angle)],
                               [axis[1] * axis[0] * (1 - np.cos(angle)) + axis[2] * np.sin(angle),
                                np.cos(angle) + axis[1]**2 * (1 - np.cos(angle)),
                                axis[1] * axis[2] * (1 - np.cos(angle)) - axis[0] * np.sin(angle)],
                               [axis[2] * axis[0] * (1 - np.cos(angle)) - axis[1] * np.sin(angle),
                                axis[2] * axis[1] * (1 - np.cos(angle)) + axis[0] * np.sin(angle),
                                np.cos(angle) + axis[2]**2 * (1 - np.cos(angle))]])

    rotated_vector = np.dot(rotation_matrix, vector)
    return rotated_vector

# Create 3D plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Define three vectors
vector1 = np.array([1, 0, 0])
vector2 = np.array([0, 1, 0])
vector3 = np.array([0, 0, 1])

# Plot the original vectors
ax.quiver(0, 0, 0, vector1[0], vector1[1], vector1[2], color='r', label='Vector 1')
ax.quiver(0, 0, 0, vector2[0], vector2[1], vector2[2], color='g', label='Vector 2')
ax.quiver(0, 0, 0, vector3[0], vector3[1], vector3[2], color='b', label='Vector 3')

# Define rotation axis and angle
rotation_axis = np.array([1, 1, 1])
rotation_angle = np.pi / 2  # 90 degrees

# Rotate vectors
rotated_vector1 = rotate_vector(vector1, rotation_axis, rotation_angle)
rotated_vector2 = rotate_vector(vector2, rotation_axis, rotation_angle)
rotated_vector3 = rotate_vector(vector3, rotation_axis, rotation_angle)

# Plot the rotated vectors
ax.quiver(0, 0, 0, rotated_vector1[0], rotated_vector1[1], rotated_vector1[2], color='r', linestyle='dashed')
ax.quiver(0, 0, 0, rotated_vector2[0], rotated_vector2[1], rotated_vector2[2], color='g', linestyle='dashed')
ax.quiver(0, 0, 0, rotated_vector3[0], rotated_vector3[1], rotated_vector3[2], color='b', linestyle='dashed')

# Set plot limits
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# Set plot labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')

# Add legend
ax.legend()

# Show the plot
plt.show()

