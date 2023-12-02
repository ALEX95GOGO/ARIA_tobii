DEBUG = False
NROWS = 1008
MARKER_FS = 120
BR_FS = 18
ACC_FS = 100
IMU_FS = 120
N_MARKERS = 7

ACC_THOLD = 10
WIN_THOLD = 0.03
MQA_THOLD = 0.7
FS_RESAMPLE = 256

WINDOW_SIZE = 20 # seconds
WINDOW_SHIFT = 1 # seconds
MIN_RESP_RATE = 3 # BPM
MAX_RESP_RATE = 45 # BPM

TIME_COLS = ['Timestamps', 'Event', 'Text', 'Color']

MOCAP_ACCEL_SD = 0.00352

TRAIN_VAL_TEST_SPLIT = [0.6, 0.2, 0.2]
TRAIN_TEST_SPLIT = [0.8, 0.2]

import matplotlib as mpl
mpl.rcParams['figure.titlesize']   = 6
mpl.rcParams['axes.titlesize']   = 6
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.labelsize']   = 6
mpl.rcParams['xtick.labelsize']  = 6
mpl.rcParams['ytick.labelsize']  = 6

LOW_HACC_FS_ID = [1, 2, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24,
                  25, 26, 27]
NO_HACC_ID = [3, 4, 5, 6]

# issues with marker data on MR conditions:
MARKER_ISSUES_MR = [12, 14, 17, 18, 26, 30]
MARKER_ISSUES_R = [12, 14, 18]
MARKER_ISSUES_M = [12, 14]
# issues with imu data on MR and L0-3 conditions:
IMU_ISSUES = [17, 21, 23, 26, 28, 30]
IMU_ISSUES_L = [15, 17, 21, 23, 26, 28]

# issues with imu data on MR:
IMU_ISSUES_MR = [17, 26, 30]

DPI = 300
FIG_FMT = 'pdf'

DATA_DIR = '/data/rqchia/ARIA/Data'
USER_DATA_DIR = '/data/rqchia/'
PROJ_DIR = '/projects/CIBCIGroup/00DataUploading/rqchia/aria_static/'
