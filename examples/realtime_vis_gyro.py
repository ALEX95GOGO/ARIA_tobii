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


fig1, ax = plt.subplots(figsize=(7, 5))
fig = plt.figure()
canvas = np.zeros((480,640))
screen = pf.screen(canvas, 'Sinusoid')

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



def degree_range(n): 
    start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
    end = np.linspace(0,180,n+1, endpoint=True)[1::]
    mid_points = start + ((end-start)/2.)
    return np.c_[start, end], mid_points
def rot_text(ang): 
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation

def generate_imu_features(df):
    features = []
    """
    'acc_x__change_quantiles__f_agg_"var"__isabs_False__qh_0.4__ql_0.2',
    'acc_x__change_quantiles__f_agg_"var"__isabs_True__qh_0.4__ql_0.2',
    'acc_x__fft_coefficient__attr_"real"__coeff_10',
    'acc_x__quantile__q_0.2', 'acc_x__quantile__q_0.3',
    'acc_y__fft_coefficient__attr_"real"__coeff_10',
    'acc_z__agg_linear_trend__attr_"stderr"__chunk_len_50__f_agg_"max"',
    'acc_z__c3__lag_1', 'acc_z__c3__lag_2', 'acc_z__c3__lag_3',
    'acc_z__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.4',
    'acc_z__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.6',
    'acc_z__change_quantiles__f_agg_"mean"__isabs_True__qh_1.0__ql_0.8',
    'acc_z__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.2',
    'acc_z__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.4',
    'acc_z__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.6',
    'acc_z__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.8',
    'acc_z__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.2',
    'acc_z__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.4',
    'acc_z__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.6',
    'acc_z__change_quantiles__f_agg_"var"__isabs_True__qh_1.0__ql_0.8',
    'acc_z__maximum', 'acc_z__skewness',
    'acc_z__spkt_welch_density__coeff_5',
    'gyro_x__fft_coefficient__attr_"real"__coeff_70',
    'gyro_y__fft_coefficient__attr_"abs"__coeff_13',
    'gyro_y__large_standard_deviation__r_0.125000000000000002',
    'gyro_z__benford_correlation',
    'gyro_z__fft_coefficient__attr_"abs"__coeff_13',
    'gyro_z__fft_coefficient__attr_"imag"__coeff_10'],
    """
    acc_x, acc_y, acc_z = df['acc_x'], df['acc_y'], df['acc_z']
    gyro_x, gyro_y, gyro_z = df['gyro_x'], df['gyro_y'], df['gyro_z']

    acc_x_features = [
        change_quantiles(acc_x, f_agg='var', isabs=False, qh=0.4, ql=0.2),
        change_quantiles(acc_x, f_agg='var', isabs=True, qh=0.4, ql=0.2),
        fft_coefficient(acc_x, agg='real', coeff=10),
        quantile(acc_x, q=0.2),
        quantile(acc_x, q=0.3),
    ]

    acc_y_features = [
        fft_coefficient(acc_y, agg='real', coeff=10),
    ]

    acc_z_features = [
        agg_linear_trend(acc_z, attr='stderr', chunk_len=50, f_agg='max'),
        c3(acc_z, 1),
        c3(acc_z, 2),
        c3(acc_z, 3),
        change_quantiles(acc_z, f_agg='mean', isabs=True, qh=1, ql=0.4),
        change_quantiles(acc_z, f_agg='mean', isabs=True, qh=1, ql=0.6),
        change_quantiles(acc_z, f_agg='mean', isabs=True, qh=1, ql=0.8),
        change_quantiles(acc_z, f_agg='var', isabs=False, qh=1, ql=0.2),
        change_quantiles(acc_z, f_agg='var', isabs=False, qh=1, ql=0.4),
        change_quantiles(acc_z, f_agg='var', isabs=False, qh=1, ql=0.6),
        change_quantiles(acc_z, f_agg='var', isabs=False, qh=1, ql=0.8),
        change_quantiles(acc_z, f_agg='var', isabs=True, qh=1, ql=0.2),
        change_quantiles(acc_z, f_agg='var', isabs=True, qh=1, ql=0.4),
        change_quantiles(acc_z, f_agg='var', isabs=True, qh=1, ql=0.6),
        change_quantiles(acc_z, f_agg='var', isabs=True, qh=1, ql=0.8),
        ('maximum', np.max(acc_z)),
        skewness(acc_z),
        spkt_welch_density(acc_z, {'coeff': 5}),
    ]

    gyro_x_features = [
        fft_coefficient(gyro_x, agg='real', coeff=70)
    ]

    gyro_y_features = [
        fft_coefficient(gyro_y, agg='abs', coeff=13),
        large_standard_deviation(gyro_y, r=0.15)
    ]

    gyro_z_features = [
        benford_correlation(gyro_z),
        fft_coefficient(gyro_z, agg='abs', coeff=13),
        fft_coefficient(gyro_z, agg='imag', coeff=10),
    ]

    def set_feature_names(features, sensor='acc_x'):
        for i, (index, val) in enumerate(features):
            index = str(index)
            features[i] = [sensor+index, val]

    set_feature_names(acc_x_features, sensor='acc_x__')
    set_feature_names(acc_y_features, sensor='acc_y__')
    set_feature_names(acc_z_features, sensor='acc_z__')
    set_feature_names(gyro_x_features, sensor='gyro_x__')
    set_feature_names(gyro_y_features, sensor='gyro_y__')
    set_feature_names(gyro_z_features, sensor='gyro_z__')

    features = acc_x_features + acc_y_features + acc_z_features \
            + gyro_x_features + gyro_y_features + gyro_z_features
    return pd.DataFrame(features)



angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False).tolist()





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

def gauge(labels=[ 1, 2, 3, 4, 5],
          colors=['red','orangered','orange','skyblue','blue'], 
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
    
    [ax.add_patch(p) for p in patches]

    
    """
    set the labels
    """

    for mid, lab in zip(mid_points, labels): 

        ax.text(0.42 * np.cos(np.radians(mid)), 0.42 * np.sin(np.radians(mid)), lab, \
            horizontalalignment='center', verticalalignment='center', fontsize=40, \
            fontweight='bold', rotation = rot_text(mid))

    """
    set the bottom banner and the title
    """
    
    r = Rectangle((-0.4,-0.1),0.8,0.1, facecolor='w', lw=2)
    ax.add_patch(r)
    

    
    ax.text(0, -0.1, title, horizontalalignment='center', \
         verticalalignment='center', fontsize=18 )

    """
    plots the arrow now
    """
    
    pos = mid_points[abs(arrow - N)]
    
    ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)), \
                 width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')
    
    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

    """
    removes frame and ticks, and makes axis equal and tight
    """
    
    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')


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
    
    #annotation = ax.annotate("50", xytext=(0,0), xy=(gyro_reading[0]*10, gyro_reading[0]*10),
    #         arrowprops=dict(arrowstyle="wedge, tail_width=0.5", color="black", shrinkA=0),
    #         bbox=dict(boxstyle="circle", facecolor="black", linewidth=2.0, ),
    #         fontsize=45, color="white", ha="center"
    #        );
    #plt.show()
    #annotation.remove()

def main():
    '''
    while True:
        asyncio.run(subscribe_to_signal())
        print(imu_data)
        # Create a real-time animated plot
        fig = plt.figure(figsize=(10, 6))
        ani = FuncAnimation(fig, animate, interval=1000)  # Update every 1 second
        plt.tight_layout()
        plt.show(block=False)
    '''
    i = 0
    while True:
        i = i + 1
        asyncio.run(subscribe_to_signal())
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

        plt.xlim(now-2,now+1)
        #plt.ylim(-3,3)
        plt.plot(x, y, c='black')
        plt.plot(x, y2, c='blue')
        plt.plot(x, y3, c='red')
        plt.legend(['accerometer_x','accelerometer_y','accelerometer_z'])
        # If we haven't already shown or saved the plot, then we need to draw the figure first...
        fig.canvas.draw()

        image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        screen.update(image)

        # let data be 10s of single axis imu sensing captured at 250Hz
        #data = np.random.random((2500, 6))
        data = np.concatenate((np.expand_dims(y,1), np.expand_dims(y2,1), np.expand_dims(y3,1),np.expand_dims(g1,1), np.expand_dims(g2,1), np.expand_dims(g3,1)), axis=1)
        #import pdb; pdb.set_trace()
        # take STD Norm
        sd_data = (data - np.mean(data, axis=0))/np.std(data, axis=0)
        # Bandpass and moving mean
        x = imu_signal_processing(sd_data)

        cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        df = pd.DataFrame(x, columns=cols)

        features_df = generate_imu_features(df)
        print(features_df)
        
        # Load the SVM model
        loaded_model = joblib.load('svm_model.joblib')

        # Create a sample input with the shape (1, 30)
        #sample_input = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]]
        features_np = np.expand_dims(np.array(features_df[1]),0)
        # Make predictions using the loaded model
        predictions = loaded_model.predict(features_np)

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
