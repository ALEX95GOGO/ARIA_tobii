import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import imufusion
import sys

# Import sensor data
data = np.genfromtxt("sensor_data.csv", delimiter=",", skip_header=1)

timestamp = data[:, 0]
gyroscope = data[:, 1:4]
accelerometer = data[:, 4:7]

# Process sensor data
ahrs = imufusion.Ahrs()
euler = np.empty((len(timestamp), 3))

# Plot sensor data
_, axes = plt.subplots(nrows=3, sharex=True)

axes[0].plot(timestamp, gyroscope[:, 0], "tab:red", label="X")
axes[0].plot(timestamp, gyroscope[:, 1], "tab:green", label="Y")
axes[0].plot(timestamp, gyroscope[:, 2], "tab:blue", label="Z")
axes[0].set_title("Gyroscope")
axes[0].set_ylabel("Degrees/s")
axes[0].grid()
axes[0].legend()

axes[1].plot(timestamp, accelerometer[:, 0], "tab:red", label="X")
axes[1].plot(timestamp, accelerometer[:, 1], "tab:green", label="Y")
axes[1].plot(timestamp, accelerometer[:, 2], "tab:blue", label="Z")
axes[1].set_title("Accelerometer")
axes[1].set_ylabel("g")
axes[1].grid()
axes[1].legend()

# Process sensor data
ahrs = imufusion.Ahrs()
#euler = numpy.empty((len(timestamp), 3))

for index in range(len(timestamp)):
    ahrs.update_no_magnetometer(gyroscope[index], accelerometer[index], 1 / 100)  # 100 Hz sample rate
    euler[index] = ahrs.quaternion.to_euler()

# Plot Euler angles
axes[2].plot(timestamp, euler[:, 0], "tab:red", label="Roll")
axes[2].plot(timestamp, euler[:, 1], "tab:green", label="Pitch")
axes[2].plot(timestamp, euler[:, 2], "tab:blue", label="Yaw")
axes[2].set_title("Euler angles")
axes[2].set_xlabel("Seconds")
axes[2].set_ylabel("Degrees")
axes[2].grid()
axes[2].legend()

plt.show(block="no_block" not in sys.argv)  # don't block when script run by CI


# Set up the figure and axes for animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def euler_to_rot_matrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]])

    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]])

    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]])

    return np.dot(R_z, np.dot(R_y, R_x))

rot_matrix = np.empty((len(timestamp), 3, 3))

def update(frame):
    ahrs.update_no_magnetometer(gyroscope[frame], accelerometer[frame], 1 / 100)
    euler[frame] = ahrs.quaternion.to_euler()
    rot_matrix[frame] = euler_to_rot_matrix(euler[frame])

    ax.cla()  # Clear previous plot
    ax.set_xlim([-1, 2])
    ax.set_ylim([-1, 2])
    ax.set_zlim([-1, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Rotational Matrix Visualization')

    # Plot the wireframe cube representing the rotation matrix
    ax.plot_wireframe(*np.indices((3, 3)), rot_matrix[frame], color='r', linewidth=2)

    return ax

ani = animation.FuncAnimation(fig, update, frames=len(timestamp), blit=False)

# Show the animated plot
plt.show()
