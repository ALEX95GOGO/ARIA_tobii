import numpy as np
from sktime.transformations.panel.rocket import MiniRocketMultivariate
# transformed_data now contains the transformed features
# You can now export or use these features as needed

import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import asyncio
import logging
import os
import dotenv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from g3pylib import connect_to_glasses
import pyformulas as pf
import matplotlib.pyplot as plt
import numpy as np
import time
from modules.digitalsignalprocessing import *
import pandas as pd
import random
from matplotlib.patches import Circle, Wedge, Rectangle
import joblib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.quiver import Quiver
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.quiver import Quiver
import imufusion
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import threading
from collections import deque
from numpy import linalg as LA
from PyQt5.QtCore import QTimer

import socket
import time
import pylsl
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import random

# SELECT DATA TO STREAM
acc = True      # 3-axis acceleration
bvp = True      # Blood Volume Pulse
gsr = True      # Galvanic Skin Response (Electrodermal Activity)
tmp = True      # Temperature

serverAddress = '127.0.0.1'
serverPort = 28000
bufferSize = 14096

#deviceID = '1451CD' # 'A02088'
#deviceID = 'A01F49'
deviceID = '6A535C'

class RealTimePlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.canvas = FigureCanvas(Figure(figsize=(10, 6)))
        self.setCentralWidget(self.canvas)
        
        # Create a 2x2 grid of subplots
        self.axes = self.canvas.figure.subplots(2, 2)

        #self.ax3d = self.axes[1, 2]  # This assumes you want the 3D plot in the bottom-right


        self.fig = self.canvas.figure
        
        # ax2 = self.fig.add_subplot(111, projection='3d', position=[0.15, -0.03, 0.3, 0.7])
        # ax2.set_xlabel('X-axis')
        # ax2.set_ylabel('Y-axis')
        # ax2.set_zlabel('Z-axis')
        # ax2.set_xlim([-1, 1])
        # ax2.set_ylim([-1, 1])
        # ax2.set_zlim([-1, 1])

        # # Pre-create quiver objects
        # quiver1 = quiver2 = quiver3 =None
        # quiver1 = ax2.quiver(0, 0, 0, 1, 0, 0, color='r', label='Glasses x')
        # quiver2 = ax2.quiver(0, 0, 0, 0, 1, 0, color='g', label='Glasses y')
        # quiver3 = ax2.quiver(0, 0, 0, 0, 0, 1, color='b', label='Glasses z')

        self.x_data, self.y_data, self.z_data = [], [], []
        self.bvp_data, self.gsr_data, self.temp_data = [], [], []
        # Assuming you're plotting X, Y, Z accelerometer data in the first three subplots
        ##for ax in self.axes.flat[:3]:
        #    ax.set_xlim(0, 100)  # Adjust the X-axis fixed range
        #    ax.set_ylim(-10, 10)  # Adjust the Y-axis limits according to expected accelerometer values
        
        self.lines = [
            #self.axes[0, 0].plot(self.x_data, label='ACC')[0],
            self.axes[0, 1].plot(self.y_data, label='BVP')[0],
            self.axes[1, 0].plot(self.z_data, label='GSR')[0],
            self.axes[1, 1].plot(self.z_data, label='temperature')[0],
        ]

        self.ax3 = self.axes[1, 0]  # Assuming ax4 is defined here for workload plotting
        self.ax4 = self.axes[1, 1]  # Assuming ax4 is defined here for workload plotting


        self.lines_acc_x = self.axes[0, 0].plot([], [], label='ACC X')[0]
        self.lines_acc_y = self.axes[0, 0].plot([], [], label='ACC Y')[0]
        self.lines_acc_z = self.axes[0, 0].plot([], [], label='ACC Z')[0]


        self.lines_bvp = self.axes[0, 1].plot(self.x_data, self.y_data, self.z_data)
        self.lines_gsr = self.axes[1, 0].plot(self.x_data, self.y_data, self.z_data)
        self.lines_temp = self.axes[1, 1].plot(self.x_data, self.y_data, self.z_data)

        # Optionally, use the fourth subplot (self.axes[1, 1]) for another type of data or leave it empty

        # Add legends to the subplots
        for ax in self.axes.flat[:4]:
            ax.legend()

    def update_acc_plot(self, data):
        # Assuming `data` is a list [x, y, z]

        print(data[0])
        self.x_data.append(data[0])
        self.y_data.append(data[1])
        self.z_data.append(data[2])
        #print(self.x_data)
        #print(self.y_data)
        #print(self.z_data)
        print(len(self.x_data))
        # If the data exceeds a certain length, start removing old data
        if len(self.x_data) >20:  # Adjust as needed
            self.x_data.pop(0)
            self.y_data.pop(0)
            self.z_data.pop(0)
        # Update each line plot with new data
        self.lines_acc_x.set_data(range(self.x_data[0].shape[0]), self.x_data[-1])
        self.lines_acc_y.set_data(range(self.y_data[0].shape[0]), self.y_data[-1])
        self.lines_acc_z.set_data(range(self.z_data[0].shape[0]), self.z_data[-1])

        #print(len(self.x_data))

        # Redraw each axis
        for ax in self.axes.flat[:3]:
            ax.relim()
            ax.autoscale_view()

        self.canvas.draw()
        self.canvas.flush_events()

        # Introduce a sleep function to increase stability
        #time.sleep(0.1)  # Adjust the sleep duration as needed


    def update_bvp_plot(self, data):
        # Assuming `data` is a single BVP value
        self.bvp_data.append(data)
        
        if len(self.bvp_data) > 100:  # Adjust as needed
            self.bvp_data.pop(0)

        # Update the BVP line plot with new data
        self.lines_bvp[0].set_data(range(len(self.bvp_data)), self.bvp_data)

        # Redraw the BVP axis
        self.axes[1, 1].relim()
        self.axes[1, 1].autoscale_view()
        
        self.canvas.draw()
        self.canvas.flush_events()
    def update_gsr_plot(self, data):
        # Assuming `data` is a single BVP value
        self.gsr_data.append(data)
        
        if len(self.gsr_data) > 100:  # Adjust as needed
            self.gsr_data.pop(0)

        # Update the BVP line plot with new data
        self.lines_gsr[0].set_data(range(len(self.gsr_data)), self.gsr_data)

        # Redraw the BVP axis
        self.axes[1, 1].relim()
        self.axes[1, 1].autoscale_view()
        
        self.canvas.draw()
        self.canvas.flush_events()
    def update_temp_plot(self, data):
        # Assuming `data` is a single BVP value
        self.temp_data.append(data)
        
        if len(self.temp_data) > 100:  # Adjust as needed
            self.temp_data.pop(0)

        # Update the BVP line plot with new data
        self.lines_temp[0].set_data(range(len(self.temp_data)), self.temp_data)

        # Redraw the BVP axis
        self.axes[1, 1].relim()
        self.axes[1, 1].autoscale_view()
        
        self.canvas.draw()
        self.canvas.flush_events()


    def update_walking_speed_plot(self, predictions):
        # Clear the current plot
        self.ax4.clear()

        # Set plot limits
        self.ax4.set_xlim(0, 1.2)  # Limits of workload levels from 1 to 4

         # Set color based on workload level
        colors = plt.cm.RdYlGn_r((4 - predictions) / 2.0)  # Inverse the color scale
        # Create horizontal bar plot
        bars = self.ax4.barh(['Speed (m/s)'], [predictions], color=[colors])

        # Set labels and titles
        #self.ax4.set_xticklabels(['1', '2', '3', '4'], fontsize=12)
        self.ax4.set_title('Walking Speed', fontsize=20)
        self.ax4.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Redraw the updated plot
        self.canvas.draw()
        self.canvas.flush_events()

    def update_workload_plot(self, predictions):
        # Clear the current plot
        self.ax3.clear()

        # Set plot limits
        self.ax3.set_xlim(0, 4)  # Limits of workload levels from 1 to 4

        # Calculate workload level based on some prediction mechanism
        workload_levels = int(1.136 * (predictions[0] - 13) - 3.908)
        workload_levels = max(min(workload_levels, 4), 1)  # Clamp the value between 1 and 4

        # Set color based on workload level
        colors = plt.cm.RdYlGn_r((4 - workload_levels) / 4.0)  # Inverse the color scale

        # Create horizontal bar plot
        bars = self.ax3.barh(['Workload'], [workload_levels], color=[colors])

        # Set labels and titles
        self.ax3.set_xticklabels(['1', '2', '3', '4'], fontsize=12)
        self.ax3.set_title('Workload Level', fontsize=20)
        self.ax3.xaxis.set_major_locator(MaxNLocator(integer=True))

        # Redraw the updated plot
        self.canvas.draw()
        self.canvas.flush_events()

    def vector_to_rotation_matrix(self, vector):
        # Normalize vector
        norm_vector = vector / np.linalg.norm(vector)
        # Create rotation object
        rotation = R.from_rotvec(np.cross([0, 0, 1], norm_vector) * np.arcsin(np.linalg.norm(np.cross([0, 0, 1], norm_vector))))
        return rotation.as_matrix()

    def update_quiver_plot(self, data):
        # Assume data is an array where the last slice contains the latest directional vector
        directional_vector = data[0, 0:3, -1]
        rotation_matrix = self.vector_to_rotation_matrix(directional_vector)

        # Update vectors
        vector1 = rotation_matrix @ np.array([1, 0, 0])
        vector2 = rotation_matrix @ np.array([0, 1, 0])
        vector3 = rotation_matrix @ np.array([0, 0, 1])

        # Update quiver objects
        self.quiver1.set_segments([[[0, 0, 0], vector1]])
        self.quiver2.set_segments([[[0, 0, 0], vector2]])
        self.quiver3.set_segments([[[0, 0, 0], vector3]])

        # Redraw the plot
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

# Simulate data update
def generate_data():

    start = 0
    #i = i + 1
    sample_rate = 100
    #import pdb; pdb.set_trace()
    #data = np.ones((6, data_length))
    imu_data = imu_update_data
    now = time.time() - start

    x = np.linspace(now-2, now, 2500)
    y = np.zeros((2500,))
    y2 = np.zeros((2500,))
    y3 = np.zeros((2500,))
    data = imu_data
    accelerations_x = [data[1]['accelerometer'][0] for data in imu_data]
    #import pdb; pdb.set_trace()
    if len(accelerations_x)>2500:
        accelerations_x=accelerations_x[-2500:]
    y[-len(accelerations_x):] = accelerations_x
    accelerations_y = [data[1]['accelerometer'][1] for data in imu_data]
    if len(accelerations_y)>2500:
        accelerations_y=accelerations_y[-2500:]
    y2[-len(accelerations_y):] = accelerations_y
    
    accelerations_z = [data[1]['accelerometer'][2] for data in imu_data]
    if len(accelerations_z)>2500:
        accelerations_z=accelerations_z[-2500:]
    y3[-len(accelerations_z):] = accelerations_z
    g1 = np.zeros((2500,))
    g2 = np.zeros((2500,))
    g3 = np.zeros((2500,))
    gyro_x = [data[1]['gyroscope'][0] for data in imu_data]
    if len(gyro_x)>2500:
        gyro_x=gyro_x[-2500:]
    g1[-len(gyro_x):] = gyro_x
    gyro_y = [data[1]['gyroscope'][1] for data in imu_data]
    if len(gyro_y)>2500:
        gyro_y=gyro_y[-2500:]
    g2[-len(gyro_y):] = gyro_y
    
    gyro_z = [data[1]['gyroscope'][2] for data in imu_data]
    if len(gyro_z)>2500:
        gyro_z=gyro_z[-2500:]
    g3[-len(gyro_z):] = gyro_z
    # let data be 10s of single axis imu sensing captured at 250Hz
    #data = np.random.random((2500, 6))
    data = np.concatenate((np.expand_dims(y,1), np.expand_dims(y2,1), np.expand_dims(y3,1),np.expand_dims(g1,1), np.expand_dims(g2,1), np.expand_dims(g3,1)), axis=1)
    data = np.expand_dims(data.transpose(1,0), 0)

    W=5
    imudata = data[0].transpose(1,0)
    T = np.zeros(int(np.floor(imudata.shape[0]/W)+1))
    zupt = np.zeros(imudata.shape[0])
    a = np.zeros((1,3))
    w = np.zeros((1,3))
    var_a = 10
    var_w = 250
    inv_a = (1/var_a)
    inv_w = (1/var_w)
    acc = imudata[:,0:3]
    gyro = imudata[:,3:6]

    i=0
    
    for k in range(0,imudata.shape[0]-W+1,W): #filter through all imu readings
        smean_a = np.mean(acc[k:k+W,:],axis=0)
        for s in range(k,k+W):
            a.put([0,1,2],acc[s,:])
            w.put([0,1,2],gyro[s,:])
            T[i] += inv_a*( (a - 9.81*smean_a/LA.norm(smean_a)).dot(( a - 9.81*smean_a/LA.norm(smean_a)).T)) #acc terms
            T[i] += inv_w*( (w).dot(w.T) )
        zupt[k:k+W].fill(T[i])
        i+=1
    zupt = zupt/W
    
    #import pdb; pdb.set_trace()
    data[:,0,zupt<0.35]=0
    data[:,2,zupt<0.35]=0
    data = data[:,:,-500:]
    x = np.arange(500)/sample_rate
    y1 = data[0,0,:]
    y2 = data[0,1,:]
    y3 = data[0,2,:] 
    acc_data =[y1,y2,y3]
    #print(acc_data)
    
    
    acc_magnitude = np.sqrt(y1[-100:]**2 + y3[-100:]**2)
    sampling_time = 0.01
    confidence = sum(data[0,3,:])/200
    walking_speed = np.sum(acc_magnitude*sampling_time)/confidence

    #acc_data = [random.randint(-2000, 2000) for _ in range(3)]  # Generate random accelerometer data
    #print(acc_data[0].shape)
    temp_data = random.randint(-20, 45)  # Generate random temperature data
    workload = random.randint(1,25)  # Generate random temperature data
    #walking_speed = random.randint(0,1)  # Generate random temperature data
    window.update_acc_plot(acc_data)
    #window.update_bvp_plot(temp_data)
    window.update_walking_speed_plot(walking_speed)
    window.update_workload_plot([workload])



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

def degree_range(n): 
    start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
    end = np.linspace(0,180,n+1, endpoint=True)[1::]
    mid_points = start + ((end-start)/2.)
    return np.c_[start, end], mid_points
def rot_text(ang): 
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation

async def subscribe_to_signal():
    async with connect_to_glasses.with_hostname(G3_HOSTNAME) as g3:
        imu_queue, unsubscribe = await g3.rudimentary.subscribe_to_imu()

        async def imu_receiver():
            count = 0
            while True:
                imu_message = await imu_queue.get()
                #print(imu_message)
                if 'accelerometer' in imu_message[1].keys():
                    imu_data.append(imu_message)  # Store the IMU data
                count += 1
                
                #if count % 300 == 0:
                #    logging.info(f"Received {count} IMU messages")
                #    logging.info(f"IMU message snapshot: {imu_message}")
                imu_queue.task_done()
            print(count)
        await g3.rudimentary.start_streams()
        receiver = asyncio.create_task(imu_receiver(), name="imu_receiver")
        await asyncio.sleep(0.5)
        #await asyncio.sleep(1.5)
        #await asyncio.sleep(1200)
        await g3.rudimentary.stop_streams()
        await imu_queue.join()
        receiver.cancel()
        await unsubscribe
    return imu_data



def gauge(labels=[6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
          colors = [
    'darkred', 'firebrick', 'indianred', 'lightcoral', 'pink',
    'lavender', 'aliceblue', 'lightskyblue', 'lightsteelblue', 'powderblue',
    'lightblue', 'skyblue', 'dodgerblue', 'deepskyblue', 'cornflowerblue', 'royalblue', 'blue', 'mediumblue', 'darkblue', 'navy'
],
          arrow="", 
          title="", 
          fname=False):     
    
    """
    some sanity checks first
    
    """
    
    N = len(labels)
    
    if arrow > N: 
        raise Exception("\n\nThe category ({}) is greated than \
        the length\nof the labels ({})".format(arrow, N)) 
 

    """
    begins the plotting
    """
    
    # Reorder the colors to resemble the viridis colormap
    viridis_order = [0, 1, 2, 5, 4, 3, 6, 9, 8, 7, 10, 13, 12, 11, 14]
    ordered_colors = [colors[i] for i in viridis_order]

    print(ordered_colors)
    #fig.subplots_adjust(0,0,2,1)

    ang_range, mid_points = degree_range(N)

    labels = labels[::-1]
    
    """
    plots the sectors and the arcs
    """
    patches = []
    for ang, c in zip(ang_range, colors): 
        # sectors
        patches.append(Wedge((0.,0.), .4,*ang, facecolor='w', lw=2 ))
        # arcs
        patches.append(Wedge((0.,0.), .4,*ang, width=0.2, facecolor=c, lw=2, alpha=0.5,))
    
    [ax3.add_patch(p) for p in patches]

    
    """
    set the labels
    """

    for mid, lab in zip(mid_points, labels): 

        ax3.text(0.42 * np.cos(np.radians(mid)), 0.42 * np.sin(np.radians(mid)), lab, \
            horizontalalignment='center', verticalalignment='center', fontsize=20, \
            fontweight='bold', rotation = rot_text(mid))

    """
    set the bottom banner and the title
    """
    
    r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
    ax3.add_patch(r)
    

    
    ax3.text(0, -0.1, title, horizontalalignment='center', \
         verticalalignment='center', fontsize=18 )

    """
    plots the arrow now
    """
    
    pos = mid_points[abs(arrow - N)]
    
    ax3.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                 width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
    
    ax3.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax3.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

    """
    removes frame and ticks, and makes axis equal and tight
    """
    
    ax3.set_frame_on(False)
    ax3.axes.set_xticks([])
    ax3.axes.set_yticks([])
    ax3.axis('equal')

'''
def update_plot(i):
    # Simulate gyro data (replace this with actual data reading)
    time_data.append(i)
    gyro_reading = [random.uniform(0.8, 1) for _ in range(num_points)]  # Replace with actual gyro data
    
    # Limit the number of data points displayed
    max_data_points = 100
    if len(time_data) > max_data_points:
        time_data.pop(0)
        gyro_data.pop(0)
    
    ax1.clear()
    ax1.set_title('Real-time Classification')
    ax1.set_xticks(angles)
    ax1.set_xticklabels(['Angle 1', 'Angle 2', 'Angle 3', 'Angle 4','Workload', 'RR'])
    ax1.plot(angles, gyro_reading)
    ax1.fill(angles, gyro_reading, alpha=0.25)
    ax1.legend()
    
    # Simulate gyro data (replace this with actual data reading)
    #time_data.append(i)
    gyro_data.append([random.uniform(-1, 1),random.uniform(-1, 1),random.uniform(-1, 1)])  # Replace with actual gyro data
    
    # Limit the number of data points displayed
    max_data_points = 100
    if len(time_data) > max_data_points:
        time_data.pop(0)
        gyro_data.pop(0)
    
    ax2.clear()
    ax2.set_title('Real-time Gyroscope Data')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Gyroscope Reading')
    ax2.plot(time_data, gyro_data, label=['Acc_x','Acc_y','Acc_z'])
    ax2.legend(loc='right')
'''
# Create ax4 for the animated horizontal bar plot


# Function to update the horizontal bar plot
def update_bar(arrow_value):
    bars4[0].set_width(arrow_value)

def normalize_vector(v):
    """ Normalize a 3D vector. """
    return v / np.linalg.norm(v)

def vector_to_rotation_matrix(vector, align_with=np.array([0, 0, 1])):
    """ Convert a 3D directional vector to a rotation matrix. """
    # Normalize the input vector and the aligning vector
    v = normalize_vector(vector)
    w = normalize_vector(align_with)

    # Calculate the rotation axis (cross product)
    axis = np.cross(w, v)
    axis_length = np.linalg.norm(axis)
    if axis_length == 0:
        # The vectors are already aligned
        return np.identity(3)

    # Normalize the axis
    axis = axis / axis_length

    # Calculate the rotation angle
    angle = np.arccos(np.dot(v, w))

    # Rodrigues' rotation formula
    skew = np.array([[0, -axis[2], axis[1]], 
                     [axis[2], 0, -axis[0]], 
                     [-axis[1], axis[0], 0]])
    rotation_matrix = np.identity(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * np.dot(skew, skew)

    return rotation_matrix

# Function to run asyncio in a separate thread
def update_data():
    #await asyncio.run(subscribe_to_signal())
    new_data = asyncio.run(subscribe_to_signal())
    # Update global data array
    # Assume the range for the sensor readings is from -10 to 10 for both sensors
    #new_data = {
    #    'gyroscope': np.random.uniform(-10, 10, (3,500)).astype(np.float32),
    #    'accelerometer': np.random.uniform(-10, 10, (3,500)).astype(np.float32)
    #}

    #new_data = np.ones((6,500), dtype=np.float32)
    global imu_update_data
    imu_update_data = new_data
    #data = np.roll(data, -1, axis=1)
    #data[:, -1] = new_data

# Function to periodically update data
def fetch_data_periodically():
    while True:
        update_data()

def update_plot(i):
    start = 0
    i = i + 1
    sample_rate = 100
    #import pdb; pdb.set_trace()
    #data = np.ones((6, data_length))
    imu_data = imu_update_data
    now = time.time() - start

    x = np.linspace(now-2, now, 2500)
    y = np.zeros((2500,))
    y2 = np.zeros((2500,))
    y3 = np.zeros((2500,))
    data = imu_data
    accelerations_x = [data[1]['accelerometer'][0] for data in imu_data]
    #import pdb; pdb.set_trace()
    if len(accelerations_x)>2500:
        accelerations_x=accelerations_x[-2500:]
    y[-len(accelerations_x):] = accelerations_x
    accelerations_y = [data[1]['accelerometer'][1] for data in imu_data]
    if len(accelerations_y)>2500:
        accelerations_y=accelerations_y[-2500:]
    y2[-len(accelerations_y):] = accelerations_y
    
    accelerations_z = [data[1]['accelerometer'][2] for data in imu_data]
    if len(accelerations_z)>2500:
        accelerations_z=accelerations_z[-2500:]
    y3[-len(accelerations_z):] = accelerations_z
    g1 = np.zeros((2500,))
    g2 = np.zeros((2500,))
    g3 = np.zeros((2500,))
    gyro_x = [data[1]['gyroscope'][0] for data in imu_data]
    if len(gyro_x)>2500:
        gyro_x=gyro_x[-2500:]
    g1[-len(gyro_x):] = gyro_x
    gyro_y = [data[1]['gyroscope'][1] for data in imu_data]
    if len(gyro_y)>2500:
        gyro_y=gyro_y[-2500:]
    g2[-len(gyro_y):] = gyro_y
    
    gyro_z = [data[1]['gyroscope'][2] for data in imu_data]
    if len(gyro_z)>2500:
        gyro_z=gyro_z[-2500:]
    g3[-len(gyro_z):] = gyro_z
    # let data be 10s of single axis imu sensing captured at 250Hz
    #data = np.random.random((2500, 6))
    data = np.concatenate((np.expand_dims(y,1), np.expand_dims(y2,1), np.expand_dims(y3,1),np.expand_dims(g1,1), np.expand_dims(g2,1), np.expand_dims(g3,1)), axis=1)
    data = np.expand_dims(data.transpose(1,0), 0)

    W=5
    imudata = data[0].transpose(1,0)
    T = np.zeros(int(np.floor(imudata.shape[0]/W)+1))
    zupt = np.zeros(imudata.shape[0])
    a = np.zeros((1,3))
    w = np.zeros((1,3))
    var_a = 10
    var_w = 250
    inv_a = (1/var_a)
    inv_w = (1/var_w)
    acc = imudata[:,0:3]
    gyro = imudata[:,3:6]

    i=0
    for k in range(0,imudata.shape[0]-W+1,W): #filter through all imu readings
        smean_a = np.mean(acc[k:k+W,:],axis=0)
        for s in range(k,k+W):
            a.put([0,1,2],acc[s,:])
            w.put([0,1,2],gyro[s,:])
            T[i] += inv_a*( (a - 9.81*smean_a/LA.norm(smean_a)).dot(( a - 9.81*smean_a/LA.norm(smean_a)).T)) #acc terms
            T[i] += inv_w*( (w).dot(w.T) )
        zupt[k:k+W].fill(T[i])
        i+=1
    zupt = zupt/W
    
    #import pdb; pdb.set_trace()
    data[:,0,zupt<0.35]=0
    data[:,2,zupt<0.35]=0
    data = data[:,:,-500:]
    x = np.arange(500)/sample_rate
    y1 = data[0,0,:]
    y2 = data[0,1,:]
    y3 = data[0,2,:] 

    ax1.clear()
    #ax1.set_xlim(now - 2, now + 1)
    ax1.plot(x, y1, c='black')
    ax1.plot(x, y2, c='blue')
    ax1.plot(x, y3, c='red')
    ax1.legend(['acclerometer_x', 'accelerometer_y', 'accelerometer_z'], loc='upper right')
    ax1.set_xlim(-1,5)
    ax1.set_ylim(-1,11)
    ax1.set_title('IMU reading', fontsize=20)
    
    # If you haven't already shown or saved the plot, then you need to draw the figure
    #plt.draw()
    #plt.pause(0.001)  # This will make the plot appear and continue the execution
    #if i%10==1:
    directional_vector = data[0, 0:3, -1]
    rotation_matrix = vector_to_rotation_matrix(directional_vector)
    vector1 = rotation_matrix @ np.array([1, 0, 0])
    vector2 = rotation_matrix @ np.array([0, 1, 0])
    vector3 = rotation_matrix @ np.array([0, 0, 1])

    quiver1.set_segments([[[0, 0, 0], vector1]])
    quiver2.set_segments([[[0, 0, 0], vector2]])
    quiver3.set_segments([[[0, 0, 0], vector3]])

    plt.draw()
    plt.pause(0.001)
    
    # Fit and transform the data
    minirocket.fit(data)
    transformed_data = minirocket.transform(data)
    print(transformed_data.shape)
    
    # Assuming x_train, y_train are already defined and linr_model.joblib is the path to your pre-trained model
    
    # Load the pre-trained model
    #linr_model = joblib.load('linr_model.joblib')
    linr_model = joblib.load('linr_model_Minh.joblib')

    # Transform x_test using PolynomialFeatures
    poly = PolynomialFeatures(degree=1)
    x_test_poly = poly.fit_transform(transformed_data)

    # Perform inference
    predictions = linr_model.predict(x_test_poly)

    # Output or process the predictions
    print("---LinearRegression---")
    print(predictions)
    #arrow = np.random.randint(2,4)
    #arrow = int(predictions[0])
    #print(arrow)

    arrow_value = int(predictions[0])-5  # Assuming this is the value you want to display

    if arrow_value>20:
        arrow_value=20
    ax4.clear()
    # Set plot limits
    ax4.set_xlim(0, 4)  # Adjust the limits as needed
    #ax4.xaxis.set_major_locator(MaxNLocator(integer=True))
    # Set tick labels after creating the bar plot
    ax4.set_xticklabels(labels, fontsize=12)

    # Set plot limits and labels
    for ax, label in zip([ax4], labels):
        #ax.set_xlim(0, 10)  # Adjust the limits as needed
        ax.set_title(f'Workload Level', fontsize=20)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    #Level = 1.136*BRDiff- 3.908;
    #NASA-TLX = 33.333*BRDiff - 98.667;
    #workload_levels = np.random.randint(1,4)
    #import pdb; pdb.set_trace()

    workload_levels = int(1.136*(predictions[0]-13)- 3.908)

    if workload_levels < 1:
        workload_levels = 1
    if workload_levels > 4:
        workload_levels = 4
    colors = plt.cm.RdYlGn_r(workload_levels / len(labels))  # Color gradient from red to green
    bars4 = ax4.barh([''], [0], color=colors)  # Use tick_label to set labels
    #update_bar(workload_levels)
    bars4[0].set_width(workload_levels)



    ax6.clear()
    # Set plot limits
    ax6.set_xlim(0, 1.5)  # Adjust the limits as needed
    #ax6.xaxis.set_major_locator(MaxNLocator())
    # Set tick labels after creating the bar plot
    #ax6.set_xticklabels(labels, fontsize=12)
    ax6.set_title(f'Walking speed', fontsize=20)
    ax6.set_xlabel(f'speed(m/s)', fontsize=10)
    # Set plot limits and labels
    #for ax, label in zip([ax6], labels):
    #    #ax.set_xlim(0, 10)  # Adjust the limits as needed
    #    ax.set_title(f'Walking speed', fontsize=20)
    #    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    #Level = 1.136*BRDiff- 3.908;
    #NASA-TLX = 33.333*BRDiff - 98.667;
    #workload_levels = np.random.randint(1,4)
    #import pdb; pdb.set_trace()
    acc_magnitude = np.sqrt(y1[-100:]**2 + y3[-100:]**2)
    sampling_time = 0.01
    velocity = np.sum(acc_magnitude*sampling_time)

    print('velocity')
    print(velocity)
    workload_levels = velocity/5
    #workload_levels = int(1.136*(predictions[0]-13)- 3.908)
    if workload_levels < 0:
        workload_levels = 0
    if workload_levels > 1.5:
        workload_levels = 1.5
    colors = plt.cm.RdYlGn_r(workload_levels / len(labels))  # Color gradient from red to green
    bars4 = ax6.barh([''], [0], color=colors)  # Use tick_label to set labels
    #update_bar(workload_levels)
    bars4[0].set_width(workload_levels)

    #gauge(title= (f"BREATHING RATE \n "), arrow = arrow_value)
    

def main():

    # Start a separate thread to fetch data
    data_thread = threading.Thread(target=fetch_data_periodically, daemon=True)
    data_thread.start()

    time.sleep(0.1)
    i = 0
    

    start = time.time()

    logging.basicConfig(level=logging.INFO)

    
    # Number of data points for the spider plot
    num_points = 6
    n_batches = 1
    n_features = 6  # (acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)
    n_window_steps = 500  # Adjust this based on your actual data
    sample_rate = 100
    # Process sensor data
    ahrs = imufusion.Ahrs()
    euler = np.empty((n_window_steps, 3))
    rot_matrix = np.empty(((n_window_steps), 3, 3))
    # Initialize quivers
    #ani = FuncAnimation(fig, update_plot, frames=np.arange(data_length))
    #plt.show()
    app = QApplication(sys.argv)
    window = RealTimePlotter()
    window.show()

    # Setup timer to update plots every 100 milliseconds
    timer = QTimer()
    timer.timeout.connect(generate_data)
    timer.start(500)

    sys.exit(app.exec_())

async def subscribe_to_signal():
    async with connect_to_glasses.with_hostname(G3_HOSTNAME) as g3:
        imu_queue, unsubscribe = await g3.rudimentary.subscribe_to_imu()

        async def imu_receiver():
            count = 0
            while True:
                imu_message = await imu_queue.get()
                #print(imu_message)
                if 'accelerometer' in imu_message[1].keys():
                    imu_data.append(imu_message)  # Store the IMU data
                count += 1
                
                #if count % 300 == 0:
                #    logging.info(f"Received {count} IMU messages")
                #    logging.info(f"IMU message snapshot: {imu_message}")
                imu_queue.task_done()
            print(count)
        await g3.rudimentary.start_streams()
        receiver = asyncio.create_task(imu_receiver(), name="imu_receiver")
        await asyncio.sleep(0.5)
        #await asyncio.sleep(1.5)
        #await asyncio.sleep(12)
        await g3.rudimentary.stop_streams()
        await imu_queue.join()
        receiver.cancel()
        await unsubscribe
    return imu_data

def connect():
    global s
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)

    print("Connecting to server")
    s.connect((serverAddress, serverPort))
    print("Connected to server\n")

    print("Devices available:")
    s.send("device_list\r\n".encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))

    print("Connecting to device")
    s.send(("device_connect " + deviceID + "\r\n").encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))

    print("Pausing data receiving")
    s.send("pause ON\r\n".encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))


def suscribe_to_data():
    if acc:
        print("Suscribing to ACC")
        s.send(("device_subscribe " + 'acc' + " ON\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))
    if bvp:
        print("Suscribing to BVP")
        s.send(("device_subscribe " + 'bvp' + " ON\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))
    if gsr:
        print("Suscribing to GSR")
        s.send(("device_subscribe " + 'gsr' + " ON\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))
    if tmp:
        print("Suscribing to Temp")
        s.send(("device_subscribe " + 'tmp' + " ON\r\n").encode())
        response = s.recv(bufferSize)
        print(response.decode("utf-8"))

    print("Resuming data receiving")
    s.send("pause OFF\r\n".encode())
    response = s.recv(bufferSize)
    print(response.decode("utf-8"))



def prepare_LSL_streaming():
    print("Starting LSL streaming")
    if acc:
        infoACC = pylsl.StreamInfo('acc','ACC',3,32,'int32','ACC-empatica_e4');
        global outletACC
        outletACC = pylsl.StreamOutlet(infoACC)
    if bvp:
        infoBVP = pylsl.StreamInfo('bvp','BVP',1,64,'float32','BVP-empatica_e4');
        global outletBVP
        outletBVP = pylsl.StreamOutlet(infoBVP)
    if gsr:
        infoGSR = pylsl.StreamInfo('gsr','GSR',1,4,'float32','GSR-empatica_e4');
        global outletGSR
        outletGSR = pylsl.StreamOutlet(infoGSR)
    if tmp:
        infoTemp = pylsl.StreamInfo('tmp','Temp',1,4,'float32','Temp-empatica_e4');
        global outletTemp
        outletTemp = pylsl.StreamOutlet(infoTemp)


def reconnect():
    print("Reconnecting...")
    connect()
    suscribe_to_data()
    stream()

def stream():
    try:
        print("Streaming...")
        while True:
            try:
                response = s.recv(bufferSize).decode("utf-8")
                #print(response)
                if "connection lost to device" in response:
                    print(response.decode("utf-8"))
                    reconnect()
                    break
                samples = response.split("\n")
                for i in range(len(samples)-1):
                    stream_type = samples[i].split()[0]
                    # if stream_type == "E4_Acc":
                    #     timestamp = float(samples[i].split()[1].replace(',','.'))
                    #     data = [int(samples[i].split()[2].replace(',','.')), int(samples[i].split()[3].replace(',','.')), int(samples[i].split()[4].replace(',','.'))]
                    #     outletACC.push_sample(data, timestamp=timestamp)
                    #     window.update_acc_plot(data)
                    #     #import pdb; pdb.set_trace()
                    if stream_type == "E4_Bvp":
                        timestamp = float(samples[i].split()[1].replace(',','.'))
                        data = float(samples[i].split()[2].replace(',','.'))
                        outletBVP.push_sample([data], timestamp=timestamp)
                        bvp_data = data
                        window.update_bvp_plot(data)
                    if stream_type == "E4_Gsr":
                        timestamp = float(samples[i].split()[1].replace(',','.'))
                        data = float(samples[i].split()[2].replace(',','.'))
                        outletGSR.push_sample([data], timestamp=timestamp)
                        gsr_data = data
                        window.update_gsr_plot(data)
                    if stream_type == "E4_Temperature":
                        timestamp = float(samples[i].split()[1].replace(',','.'))
                        data = float(samples[i].split()[2].replace(',','.'))
                        outletTemp.push_sample([data], timestamp=timestamp)
                        temp_data = data
                        window.update_temp_plot(data)
                    #import pdb; pdb.set_trace()
                    #window.update_plot(data)
                #time.sleep(1)
            except socket.timeout:
                print("Socket timeout")
                reconnect()
                break
    except KeyboardInterrupt:
        print("Disconnecting from device")
        s.send("device_disconnect\r\n".encode())
        s.close()




if __name__ == "__main__":
    

    G3_HOSTNAME = 'tg03b-080200018761'
    #G3_HOSTNAME = 'tg03b-080200027081'
    # Initialize a list to store the IMU data for plotting
    imu_data = []

    # Initialize empty lists for gyro data
    time_data = []
    gyro_data = []

    fig, ((ax1, ax2, ax3), (ax4,ax5,ax6)) = plt.subplots(2, 3, figsize=(18, 12))
    # Specify ax3 for tight layout
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # Adjust rect as needed
    # Hide the original ax4 to ax6
    ax4.axis('off')
    ax5.axis('off')
    ax6.axis('off')
    ax4 = fig.add_subplot(325)  # You may need to adjust the position and size accordingly
    
    
    # ax2 = fig.add_subplot(111, projection='3d', position=[0.37, 0.37, 0.35, 0.65])
    # ax2.set_xlabel('X-axis')
    # ax2.set_ylabel('Y-axis')
    # ax2.set_zlabel('Z-axis')
    # ax2.set_xlim([-1, 1])
    # ax2.set_ylim([-1, 1])
    # ax2.set_zlim([-1, 1])
    
    # # Pre-create quiver objects
    # quiver1 = quiver2 = quiver3 =None
    # quiver1 = ax2.quiver(0, 0, 0, 1, 0, 0, color='r', label='Glasses x')
    # quiver2 = ax2.quiver(0, 0, 0, 0, 1, 0, color='g', label='Glasses y')
    # quiver3 = ax2.quiver(0, 0, 0, 0, 0, 1, color='b', label='Glasses z')
    # # Turn off auto-scaling
    # ax2.set_autoscale_on(False)

    # # Draw legend once
    # ax2.legend()


    # Initialize MiniRocketMultivariate
    minirocket = MiniRocketMultivariate()
    
    # Initialize data array
    data_length = 100
    # Initialize the horizontal bar plots
    labels = ['Very Low', 'Low', 'Medium', 'High']
    dotenv.load_dotenv()

    # Start a separate thread to fetch data
    data_thread = threading.Thread(target=fetch_data_periodically, daemon=True)
    data_thread.start()

    time.sleep(0.1)
    i = 0
    

    start = time.time()

    logging.basicConfig(level=logging.INFO)

    
    # Number of data points for the spider plot
    num_points = 6
    n_batches = 1
    n_features = 6  # (acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)
    n_window_steps = 500  # Adjust this based on your actual data
    sample_rate = 100
    # Process sensor data
    ahrs = imufusion.Ahrs()
    euler = np.empty((n_window_steps, 3))
    rot_matrix = np.empty(((n_window_steps), 3, 3))
    # Initialize quivers
    #ani = FuncAnimation(fig, update_plot, frames=np.arange(data_length))
    #plt.show()

    connect()

    time.sleep(1)
    suscribe_to_data()
    prepare_LSL_streaming()

    time.sleep(1)

    app = QApplication(sys.argv)
    window = RealTimePlotter()
    window.show()

    # Setup timer to update plots every 100 milliseconds
    timer = QTimer()
    timer.timeout.connect(generate_data)
    timer.start(100)

    stream()

    sys.exit(app.exec_())

    #main()


