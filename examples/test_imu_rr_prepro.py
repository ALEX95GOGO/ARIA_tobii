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
                #import pdb; pdb.set_trace()
                if 'accelerometer' in imu_message[1].keys():
                    imu_data.append(imu_message)  # Store the IMU data
                #import pdb; pdb.set_trace()
                count += 1
                if count % 300 == 0:
                    logging.info(f"Received {count} IMU messages")
                    logging.info(f"IMU message snapshot: {imu_message}")
                imu_queue.task_done()

        await g3.rudimentary.start_streams()
        receiver = asyncio.create_task(imu_receiver(), name="imu_receiver")
        await asyncio.sleep(0.5)
        await g3.rudimentary.stop_streams()
        await imu_queue.join()
        receiver.cancel()
        await unsubscribe

def gauge(labels=[ 1, 2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19],
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

if __name__ == '__main__':
    fig, ((ax1, ax2, ax3), (ax4,ax5,ax6)) = plt.subplots(2, 3, figsize=(18, 15))
        # Specify ax3 for tight layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect as needed
    # Hide the original ax4 to ax6
    ax4.axis('off')
    ax5.axis('off')
    ax6.axis('off')
    ax4 = fig.add_subplot(325)  # You may need to adjust the position and size accordingly

    # Initialize the horizontal bar plots
    labels = ['Very Low', 'Low', 'Medium', 'High']
    

    

    # Adjust layout
    #plt.tight_layout()

    # Show the plot
    #plt.show()
    #import pdb; pdb.set_trace()
    #canvas = np.zeros((480,640))
    #screen = pf.screen(canvas, 'RawIMU')

    start = time.time()

    logging.basicConfig(level=logging.INFO)

    G3_HOSTNAME = 'tg03b-080200018761'
    #G3_HOSTNAME = 'tg03b-080200027081'
    # Initialize a list to store the IMU data for plotting
    imu_data = []

    # Initialize empty lists for gyro data
    time_data = []
    gyro_data = []

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

    while True:
        # Sample data (replace this with your actual data)
        # Assuming the shape is (n_batches, n_features, n_window_steps)
        # For simplicity, I'm generating random data


        data = np.random.rand(n_batches, n_features, n_window_steps)

        x = np.arange(500)/sample_rate
        y1 = data[0,0,:]
        y2 = data[0,1,:]
        y3 = data[0,2,:] + 9 

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
        plt.draw()
        plt.pause(0.001)  # This will make the plot appear and continue the execution


        ax2 = fig.add_subplot(111, projection='3d', position=[0.35, 0.35, 0.3, 0.7])

        # Define three vectors
        vector1 = np.array([1, 0, 0])
        vector2 = np.array([0, 1, 0])
        vector3 = np.array([0, 0, 1])
        #import pdb; pdb.set_trace()
        ahrs.update_no_magnetometer(data[0,0:3,100], data[0,3:6,100], 1 / 100)
        euler[100] = ahrs.quaternion.to_euler()
        rot_matrix[100] = euler_to_rot_matrix(euler[100])
        
        vector1 = rot_matrix[100] * vector1
        vector2 = rot_matrix[100] * vector2
        vector3 = rot_matrix[100] * vector3

        # Plot the original vectors
        ax2.quiver(0, 0, 0, vector1[0], vector1[1], vector1[2], color='r', label='Glasses x')
        ax2.quiver(0, 0, 0, vector2[0], vector2[1], vector2[2], color='g', label='Glasses y')
        ax2.quiver(0, 0, 0, vector3[0], vector3[1], vector3[2], color='b', label='Glasses z')

        # Define rotation axis and angle
        rotation_axis = np.array([1, 1, 1])
        rotation_angle = np.pi / 2  # 90 degrees

        # Rotate vectors
        rotated_vector1 = rotate_vector(vector1, rotation_axis, rotation_angle)
        rotated_vector2 = rotate_vector(vector2, rotation_axis, rotation_angle)
        rotated_vector3 = rotate_vector(vector3, rotation_axis, rotation_angle)

        # Plot the rotated vectors
        #ax2.quiver(0, 0, 0, rotated_vector1[0], rotated_vector1[1], rotated_vector1[2], color='r', linestyle='dashed')
        #ax2.quiver(0, 0, 0, rotated_vector2[0], rotated_vector2[1], rotated_vector2[2], color='g', linestyle='dashed')
        #ax2.quiver(0, 0, 0, rotated_vector3[0], rotated_vector3[1], rotated_vector3[2], color='b', linestyle='dashed')

        # Set plot limits
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([-1, 1])
        ax2.set_zlim([-1, 1])

        # Set plot labels
        ax2.set_xlabel('X-axis')
        ax2.set_ylabel('Y-axis')
        ax2.set_zlabel('Z-axis')

        # Add legend
        ax2.legend()
        plt.draw()
        plt.pause(0.001)

        # Initialize MiniRocketMultivariate
        minirocket = MiniRocketMultivariate()

        # Fit and transform the data
        minirocket.fit(data)

        transformed_data = minirocket.transform(data)
        print(transformed_data.shape)

        # Assuming x_train, y_train are already defined and linr_model.joblib is the path to your pre-trained model

        # Load the pre-trained model
        linr_model = joblib.load('linr_model.joblib')

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

        arrow_value = int(predictions[0])  # Assuming this is the value you want to display
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


        workload_levels = np.random.randint(1,4)
        colors = plt.cm.RdYlGn_r(workload_levels / len(labels))  # Color gradient from red to green
        bars4 = ax4.barh([''], [0], color=colors)  # Use tick_label to set labels
        update_bar(workload_levels)
        gauge(title= (f"BREATHING RATE \n "), arrow = arrow_value)
        
        plt.pause(0.2) 


'''

    def main():
        
        while True:
            asyncio.run(subscribe_to_signal())
            print(imu_data)
            # Create a real-time animated plot
            fig = plt.figure(figsize=(10, 6))
            ani = FuncAnimation(fig, animate, interval=1000)  # Update every 1 second
            plt.tight_layout()
            plt.show(block=False)
        
        i = 0
        import pdb; pdb.set_trace()
        while True:
            i = i + 1
            #asyncio.run(subscribe_to_signal())
            now = time.time() - start

            x = np.linspace(now-2, now, 2500)
            y = np.zeros((2500,))
            y2 = np.zeros((2500,))
            y3 = np.zeros((2500,))
            accelerations_x = [data[1]['accelerometer'][0] for data in imu_data]
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

            
            ax1.set_xlim(now - 2, now + 1)
            ax1.plot(x, y, c='black')
            ax1.plot(x, y2, c='blue')
            ax1.plot(x, y3, c='red')
            ax1.legend(['accerometer_x', 'accelerometer_y', 'accelerometer_z'])
            
            # If you haven't already shown or saved the plot, then you need to draw the figure
            plt.draw()
            plt.pause(0.001)  # This will make the plot appear and continue the execution


            image = np.fromstring(fig2.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            image = image.reshape(fig2.canvas.get_width_height()[::-1] + (3,))

            screen.update(image)

            # let data be 10s of single axis imu sensing captured at 250Hz
            #data = np.random.random((2500, 6))
            data = np.concatenate((np.expand_dims(y,1), np.expand_dims(y2,1), np.expand_dims(y3,1),np.expand_dims(g1,1), np.expand_dims(g2,1), np.expand_dims(g3,1)), axis=1)
            #import pdb; pdb.set_trace()
            # take STD Norm
            sd_data = (data - np.mean(data, axis=0))/np.std(data, axis=0)
            # Bandpass and moving mean
            x = imu_signal_processing(sd_data)

            #cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
            #df = pd.DataFrame(x, columns=cols)

            #features_df = generate_imu_features(df)
            #print(features_df)

            #n_batches = 1
            #n_features = 6  # (acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)
            #n_window_steps = 500  # Adjust this based on your actual data

            #data = np.random.rand(n_batches, n_features, n_window_steps)
            data = np.swapaxis(data, 0,1)
            data = np.expand_dims(data,0)
            # Initialize MiniRocketMultivariate
            minirocket = MiniRocketMultivariate()

            # Fit and transform the data
            minirocket.fit(data)

            transformed_data = minirocket.transform(data)
            
            # Load the pre-trained model
            linr_model = joblib.load('linr_model.joblib')

            # Transform x_test using PolynomialFeatures
            poly = PolynomialFeatures(degree=1)
            x_test_poly = poly.fit_transform(transformed_data)

            # Perform inference
            predictions = linr_model.predict(x_test_poly)

            # Output or process the predictions
            print("---LinearRegression---")
            print(predictions)
            # Configure the first subplot as a polar plot
            
            #arrow = np.random.randint(2,4)
            arrow = int(predictions[0])
            print(arrow)
            gauge(title= (f"WORKLOAD SCORE \n "), arrow = arrow)
            print(arrow)
            plt.pause(0.2) 
            #ax.clear()
            
            # Function to update the plot with new data





    if __name__ == "__main__":
        dotenv.load_dotenv()
        main()

'''

