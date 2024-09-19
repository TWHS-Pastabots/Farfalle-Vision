import constants
import utils
import time
import numpy as np
from cscore import CameraServer as cs
from cscore import VideoSource as vs
import cv2
import robotpy_apriltag as rptag
from ntcore import NetworkTableInstance as nt
from scipy.spatial.transform import Rotation
import ntcore

# Initialize network tables
vision_table = utils.launch_network_table()

# Initialize all cameras
cam1_input_stream = utils.launch_cam(constants.CAM1_NAME, 1)
cam2_input_stream = utils.launch_cam(constants.CAM2_NAME, 2)
cam3_input_stream = utils.launch_cam(constants.CAM3_NAME, 3)
cam4_input_stream = utils.launch_cam(constants.CAM4_NAME, 4)

# Sleep for a bit to ensure previous steps have finished
time.sleep(0.1)

# Set up output stream
output_stream = utils.get_output_stream()

# Remove scientific notation
np.set_printoptions(suppress = True)

# Get AprilTag detector
tag_detector = utils.get_tag_detector()

# Sleep for a bit again to ensure previous steps have finished
time.sleep(0.1)

# Main loop for stream processing. Loop can be set to true because for match/testing it should keep running until robot is off
while(true):

    # Cam1 processing, can add gamepiece detection if desired
    try:
        frame = utils.get_frame(cam1_input_stream, output_stream)
        utils.find_tags(frame, tag_detector, 1, vision_table)
    except:
        print("Error with cam 1 processing")

    # Cam2 processing, can add gamepiece detection if desired
    try:
        frame = utils.get_frame(cam2_input_stream, output_stream)
        utils.find_tags(frame, tag_detector, 2, vision_table)
    except:
        print("Error with cam 2 processing")

    # Cam3 processing, can add gamepiece detection if desired
    try:
         frame = utils.get_frame(cam3_input_stream, output_stream)
         utils.find_tags(frame, tag_detector, 3, vision_table)
    except:
         print("Error with cam 3 processing")

    # Cam3 processing, can add gamepiece detection if desired
    try:
         frame = utils.get_frame(cam4_input_stream, output_stream)
         utils.find_tags(frame, tag_detector, 4, vision_table)
    except:
         print("Error with cam 4 processing")
