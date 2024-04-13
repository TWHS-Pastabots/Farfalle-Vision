import time
import numpy as np
from cscore import CameraServer as cs
from cscore import VideoSource as vs
import cv2
import robotpy_apriltag as rptag
from ntcore import NetworkTableInstance as nt
from scipy.spatial.transform import Rotation
import ntcore

# Set up network tables
inst = nt.getDefault()
inst.startClient4("visiontwhs")
inst.setServerTeam(9418)
inst.startDSClient()
vision_table = inst.getTable('Vision')
time.sleep(0.5)

width = 640
height = 480
fps = 15

# Set up cameras
usb1 = cs.startAutomaticCapture(name = "Launcher Cam", path = "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.1:1.0-video-index0")
usb1.setConnectionStrategy(vs.ConnectionStrategy.kConnectionKeepOpen)
usb1.setResolution(width, height)
usb1.setFPS(fps)

usb2 = cs.startAutomaticCapture(name = "Left Cam", path = "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.2:1.0-video-index0")
usb2.setConnectionStrategy(vs.ConnectionStrategy.kConnectionKeepOpen)
usb2.setResolution(width, height)
usb2.setFPS(fps)

usb3 = cs.startAutomaticCapture(name = "Right Cam", path = "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.4:1.0-video-index0")
usb3.setConnectionStrategy(vs.ConnectionStrategy.kConnectionKeepOpen)
usb3.setResolution(width, height)
usb3.setFPS(fps)

usb4 = cs.startAutomaticCapture(name = "Intake Cam", path = "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.3:1.0-video-index0")
usb4.setConnectionStrategy(vs.ConnectionStrategy.kConnectionKeepOpen)
usb4.setResolution(width, height)
usb4.setFPS(fps)

cam1_input_stream = cs.getVideo(camera = usb1)
cam2_input_stream = cs.getVideo(camera = usb2)
cam3_input_stream = cs.getVideo(camera = usb3)
cam4_input_stream = cs.getVideo(camera = usb4)

time.sleep(0.1)

output_stream = cs.putVideo(name = 'Cam Stream', width = width,  height = height)
img = np.zeros(shape = (height, width, 3), dtype = np.uint8)

# Remove scientific notation
np.set_printoptions(suppress = True)

# Set up tag detectors + estimators
detector = rptag.AprilTagDetector()
detector.addFamily("tag36h11")

DETECTION_MARGIN_THRESHOLD = 90

# Camera/tag parameters
tag_size_m = 0.1651
fx = 699.3778103158814
fy = 677.7161226393544
cx = 345.6059345433618
cy = 207.12741326228522

# Intrinsics matrix
intrinsics_mat = np.array(
    [[fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]],
    dtype = np.float64
)

# Camera distortion
distortions = np.array(
    [0.14382207979312617,
    -0.9851192814987014,
    -0.018168751047242335,
    0.011034504043795105,
    1.9833437176538498],
    dtype = np.float64
)

# Ideal Corners
t = tag_size_m / 2
obj_pts = np.array(
    [[-t, -t, 0],
    [t, -t, 0],
    [t, t, 0],
    [-t, t, 0]],
    dtype=np.float32)

time.sleep(0.1)

# Detects and estimates pose using solvePnP() for apriltags for cam1
def cam1TagDetect():
        # Grab frame from cam1 input stream
        cam1_frame_time, cam1_input_img = cam1_input_stream.grabFrame(img)

        if cam1_frame_time == 0: # If frame time is zero then there was no time between the last frame so no new data
            output_stream.notifyError(cam1_input_stream.getError())

        # Setting up tag info lists     
        serialized_tags_list = []

        # Grayscale frame + detect
        gray_img = cv2.cvtColor(cam1_input_img, cv2.COLOR_BGR2GRAY)
        tag_info = detector.detect(gray_img)

        # Filter out bad detections (low decision margin + out of bounds IDs)
        filter_tags = [tag for tag in tag_info if tag.getDecisionMargin() > DETECTION_MARGIN_THRESHOLD]
        filter_tags = [tag for tag in filter_tags if ((tag.getId() > 0) & (tag.getId() < 17))]                  

        # Process detections
        for tag in filter_tags:

            # Get corners of AprilTag
            corners = np.array(
                [[tag.getCorner(0).x, tag.getCorner(0).y],
                [tag.getCorner(1).x, tag.getCorner(1).y],
                [tag.getCorner(2).x, tag.getCorner(2).y],
                [tag.getCorner(3).x, tag.getCorner(3).y]],
                dtype = np.float64
            )

            # Get tag space rotation vector and translation vector with solvePnP()
            _, r_vec, t_vec = cv2.solvePnP(
                objectPoints = obj_pts,
                imagePoints = corners,
                cameraMatrix = np.asarray(intrinsics_mat), 
                distCoeffs = np.asarray(distortions),
                flags = cv2.SOLVEPNP_SQPNP
            )
    
            # Convert tag space to camera space and set up transform
            r_mat = cv2.Rodrigues(r_vec)[0]

            # T = -r^T * t
            T_cs = -np.matrix(r_mat).T * np.matrix(t_vec)
            T_nt = [
                float(T_cs[0]),
                float(T_cs[1]),
                float(T_cs[2])
            ]

            R_q = Rotation.from_matrix(r_mat).as_quat()
            R_nt = [
                float(R_q[0]),
                float(R_q[1]),
                float(R_q[2]),
                float(R_q[3]),
            ]
            
            # Serialized tag information
            tag_serial_string = str(tag.getId()) + " " + str(T_nt[0]) + " " + str(T_nt[1]) + " " + str(T_nt[2])+ " "
            tag_serial_string  += str(R_nt[0]) + " " + str(R_nt[1]) + " " + str(R_nt[2]) + " " + str(R_nt[3]) + " "
            tag_serial_string += str(ntcore._now())       
            serialized_tags_list.insert(0, tag_serial_string)

            # Pop list in case it get too big to avoid memory issues
            if len(serialized_tags_list) > 10:
                serialized_tags_list.pop()
        
        # Send serialized tags over network tables
        vision_table.putStringArray("Serialized Tags", serialized_tags_list)

# Detects and estimates pose using solvePnP() for apriltags for cam2
def cam2TagDetect():
        # Grab frame from cam2 input stream
        cam2_frame_time, cam2_input_img = cam2_input_stream.grabFrame(img)

        if cam2_frame_time == 0: # If frame time is zero then there was no time between the last frame so no new data
            output_stream.notifyError(cam2_input_stream.getError())

        # Setting up tag info lists     
        serialized_tags_list = []

        # Grayscale frame + detect
        gray_img = cv2.cvtColor(cam2_input_img, cv2.COLOR_BGR2GRAY)
        tag_info = detector.detect(gray_img)

        # Filter out bad detections (low decision margin + out of bounds IDs)
        filter_tags = [tag for tag in tag_info if tag.getDecisionMargin() > DETECTION_MARGIN_THRESHOLD]
        filter_tags = [tag for tag in filter_tags if ((tag.getId() > 0) & (tag.getId() < 17))]                  

        # Process detections
        for tag in filter_tags:

            # Get corners of AprilTag
            corners = np.array(
                [[tag.getCorner(0).x, tag.getCorner(0).y],
                [tag.getCorner(1).x, tag.getCorner(1).y],
                [tag.getCorner(2).x, tag.getCorner(2).y],
                [tag.getCorner(3).x, tag.getCorner(3).y]],
                dtype = np.float64
            )

            # Get tag space rotation vector and translation vector with solvePnP()
            _, r_vec, t_vec = cv2.solvePnP(
                objectPoints = obj_pts,
                imagePoints = corners,
                cameraMatrix = np.asarray(intrinsics_mat), 
                distCoeffs = np.asarray(distortions),
                flags = cv2.SOLVEPNP_SQPNP
            )
    
            # Convert tag space to camera space and set up transform
            r_mat = cv2.Rodrigues(r_vec)[0]

            # T = -r^T * t
            T_cs = -np.matrix(r_mat).T * np.matrix(t_vec)
            T_nt = [
                float(T_cs[0]),
                float(T_cs[1]),
                float(T_cs[2])
            ]

            R_q = Rotation.from_matrix(r_mat).as_quat()
            R_nt = [
                float(R_q[0]),
                float(R_q[1]),
                float(R_q[2]),
                float(R_q[3]),
            ]
            
            # Serialized tag information
            tag_serial_string = str(tag.getId()) + " " + str(T_nt[0]) + " " + str(T_nt[1]) + " " + str(T_nt[2])+ " "
            tag_serial_string  += str(R_nt[0]) + " " + str(R_nt[1]) + " " + str(R_nt[2]) + " " + str(R_nt[3]) + " "
            tag_serial_string += str(ntcore._now())       
            serialized_tags_list.insert(0, tag_serial_string)

            # Pop list in case it get too big to avoid memory issues
            if len(serialized_tags_list) > 10:
                serialized_tags_list.pop()
        
        # Send serialized tags over network tables
        vision_table.putStringArray("Serialized Tags 2", serialized_tags_list)

# Detects and estimates pose using solvePnP() for apriltags for cam3
def cam3TagDetect():
        # Grab frame from cam3 input stream
        cam3_frame_time, cam3_input_img = cam3_input_stream.grabFrame(img)

        if cam3_frame_time == 0: # If frame time is zero then there was no time between the last frame so no new data
            output_stream.notifyError(cam3_input_stream.getError())

        # Setting up tag info lists     
        serialized_tags_list = []

        # Grayscale frame + detect
        gray_img = cv2.cvtColor(cam3_input_img, cv2.COLOR_BGR2GRAY)
        tag_info = detector.detect(gray_img)

        # Filter out bad detections (low decision margin + out of bounds IDs)
        filter_tags = [tag for tag in tag_info if tag.getDecisionMargin() > DETECTION_MARGIN_THRESHOLD]
        filter_tags = [tag for tag in filter_tags if ((tag.getId() > 0) & (tag.getId() < 17))]                  

        # Process detections
        for tag in filter_tags:

            # Get corners of AprilTag
            corners = np.array(
                [[tag.getCorner(0).x, tag.getCorner(0).y],
                [tag.getCorner(1).x, tag.getCorner(1).y],
                [tag.getCorner(2).x, tag.getCorner(2).y],
                [tag.getCorner(3).x, tag.getCorner(3).y]],
                dtype = np.float64
            )

            # Get tag space rotation vector and translation vector with solvePnP()
            _, r_vec, t_vec = cv2.solvePnP(
                objectPoints = obj_pts,
                imagePoints = corners,
                cameraMatrix = np.asarray(intrinsics_mat), 
                distCoeffs = np.asarray(distortions),
                flags = cv2.SOLVEPNP_SQPNP
            )
    
            # Convert tag space to camera space and set up transform
            r_mat = cv2.Rodrigues(r_vec)[0]

            # T = -r^T * t
            T_cs = -np.matrix(r_mat).T * np.matrix(t_vec)
            T_nt = [
                float(T_cs[0]),
                float(T_cs[1]),
                float(T_cs[2])
            ]

            R_q = Rotation.from_matrix(r_mat).as_quat()
            R_nt = [
                float(R_q[0]),
                float(R_q[1]),
                float(R_q[2]),
                float(R_q[3]),
            ]
            
            # Serialized tag information
            tag_serial_string = str(tag.getId()) + " " + str(T_nt[0]) + " " + str(T_nt[1]) + " " + str(T_nt[2])+ " "
            tag_serial_string  += str(R_nt[0]) + " " + str(R_nt[1]) + " " + str(R_nt[2]) + " " + str(R_nt[3]) + " "
            tag_serial_string += str(ntcore._now())       
            serialized_tags_list.insert(0, tag_serial_string)

            # Pop list in case it get too big to avoid memory issues
            if len(serialized_tags_list) > 10:
                serialized_tags_list.pop()
        
        # Send serialized tags over network tables
        vision_table.putStringArray("Serialized Tags 3", serialized_tags_list)

def main():
    while True:

        # For each camera, run apriltag processing method 
        # If it fails, an empty array is published to network tables so that the rio doesn't use old tag detections
        try:
            cam1TagDetect()
        except:
            vision_table.putStringArray("Serialized Tags", [])

        try:
            cam2TagDetect()
        except:
            vision_table.putStringArray("Serialized Tags 2", [])
        
        try:
            cam3TagDetect()
        except:
             vision_table.putStringArray("Serialized Tags 3", [])
main()