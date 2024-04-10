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

# usb2 = cs.startAutomaticCapture(name = "Left Cam", path = "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.2:1.0-video-index0")
# usb2.setConnectionStrategy(vs.ConnectionStrategy.kConnectionKeepOpen)
# usb2.setResolution(width, height)
# usb2.setFPS(fps)

# usb3 = cs.startAutomaticCapture(name = "Right Cam", path = "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.4:1.0-video-index0")
# usb3.setConnectionStrategy(vs.ConnectionStrategy.kConnectionKeepOpen)
# usb3.setResolution(width, height)
# usb3.setFPS(fps)

# usb4 = cs.startAutomaticCapture(name = "Intake Cam", path = "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.3:1.0-video-index0")
# usb4.setConnectionStrategy(vs.ConnectionStrategy.kConnectionKeepOpen)
# usb4.setResolution(width, height)
# usb4.setFPS(fps)

cam1_input_stream = cs.getVideo(camera = usb1)
# cam2_input_stream = cs.getVideo(camera = usb2)
# cam3_input_stream = cs.getVideo(camera = usb3)
# cam4_input_stream = cs.getVideo(camera = usb4)

time.sleep(0.1)

output_stream = cs.putVideo(name = 'Cam Stream', width = width,  height = height)
img = np.zeros(shape = (height, width, 3), dtype = np.uint8)

# Remove scientific notation
np.set_printoptions(suppress = True)

# Set up tag detectors + estimators
detector = rptag.AprilTagDetector()
detector.addFamily("tag36h11")

DETECTION_MARGIN_THRESHOLD = 90

tag_size_m = 0.1651
fx = 699.3778103158814
fy = 677.7161226393544
cx = 345.6059345433618
cy = 207.12741326228522

intrinsics_mat = [
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
]

distortions = [ 
    0.14382207979312617,
    -0.9851192814987014,
    -0.018168751047242335,
    0.011034504043795105,
    1.9833437176538498
]

t = tag_size_m / 2
obj_pts = np.array(
    [[[-t, -t, 0], \
    [t, -t, 0], \
    [t, t, 0], \
    [-t, t, 0]]], \
    dtype=np.float32)

# Tagsize (m), fx, fy, cx, cy
#tag_estimator_conf = rptag.AprilTagPoseEstimator.Config(tag_size_m, fx, fy, cx, cy)
#tag_estimator = rptag.AprilTagPoseEstimator(tag_estimator_conf)

time.sleep(0.1)

# Detects apriltags for cam1
def cam1TagDetect():
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

            # object space rotation vector and translation vector
            _, r_vec, t_vec = cv2.solvePnP(
                obj_pts,
                np.array(
                    tag.getCorners(), 
                    dtype=np.float64
                ), 
                cameraMatrix = np.array(intrinsics_mat), 
                distCoeffs = np.array(distortions),
                flags = cv2.SOLVEPNP_SQPNP
            )
    
            # convert object space to camera space
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
            
            #Serialized tag information
            tag_serial_string = tag.getId() + " " + T_nt[0] + " " + T_nt[1] + " " + T_nt[2] + " "
            tag_serial_string  += R_nt[0] + " " + R_nt[1] + " " + R_nt[2] + " " + R_nt[3] + " "
            tag_serial_string += ntcore._now()       
            
            serialized_tags_list.insert(0, tag_serial_string)

            #pop list in case they get too big to avoid memory issues
            if len(serialized_tags_list) > 10:
                serialized_tags_list.pop()

        vision_table.putStringArray("Serialized Tags", serialized_tags_list)

# def cam2TagDetect():
#         cam2_frame_time, cam2_input_img = cam2_input_stream.grabFrame(img)

#         if cam2_frame_time == 0:
#             output_stream.notifyError(cam2_input_stream.getError())

#         # Setting up tag info lists           
#         x_list = []
#         y_list = []
#         z_list = []
#         yaw_list = []
#         id_list = []
#         timestamp_list = []
#         # x_euler_list = []
#         # y_euler_list = []
#         # z_euler_list = []
#         # bestID = -1
#         # bestX = -1
#         # bestY = -1
#         # bestZ = -1

#         # Grayscale frame + detect
#         gray_img = cv2.cvtColor(cam2_input_img, cv2.COLOR_BGR2GRAY)
#         tag_info = detector.detect(gray_img)

#         # Filter out bad detections (low decision margin + out of bounds IDs)
#         filter_tags = [tag for tag in tag_info if tag.getDecisionMargin() > DETECTION_MARGIN_THRESHOLD]
#         filter_tags = [tag for tag in filter_tags if ((tag.getId() > 0) & (tag.getId() < 17))]
        
#         # Setting up best tag stuff
#         # if len(filter_tags) > 0:
#         #     bestTag = filter_tags[0]
#         #     for tag in filter_tags:
#         #         if tag.getDecisionMargin() > bestTag.getDecisionMargin():
#         #             bestTag = tag
#         #     bestID = bestTag.getId()
#         #     bestTagPos = tag_estimator.estimate(bestTag)
#         #     bestX = bestTagPos.getX()
#         #     bestY = bestTagPos.getY()
#         #     bestZ = bestTagPos.getZ()
#         #     vision_table.putNumber("Best Timestamp", ntcore._now())                    

#         # Send detections info over network tables
#         for tag in filter_tags:
#             tag_id = tag.getId()
#             # homography = tag.getHomographyMatrix()
#             # euler_list = rotationMatrixToEulerAngles(homography)
#             tag_pos = tag_estimator.estimate(tag)
#             x_list.insert(0, tag_pos.X())
#             y_list.insert(0, tag_pos.Y())
#             z_list.insert(0, tag_pos.Z())
#             yaw_list.insert(0, tag_pos.rotation().Z())
#             timestamp_list.insert(0, ntcore._now())
#             # x_list.insert(0, (center.y))
#             # y_list.insert(0, (center.x))
#             id_list.insert(0, tag_id)
#             # z_euler_list.insert(0, euler_list[2])
#             # x_euler_list.insert(euler_list[0] * 1000)
#             # y_euler_list.insert(euler_list[1] * 1000)

#             #pop lists in case they get too big to avoid memory issues
#             if len(x_list) > 10:
#                 x_list.pop()

#             if len(y_list) > 10:
#                 y_list.pop()
            
#             if len(z_list) > 10:
#                 z_list.pop()
            
#             if len(yaw_list) > 10:
#                 yaw_list.pop()

#             if len(id_list) > 10:
#                 id_list.pop()

#             if len(timestamp_list) > 10:
#                 timestamp_list.pop()
#             # if len(z_euler_list) > 10:
#             #     z_euler_list.pop()

#         vision_table.putNumberArray("IDs 2", id_list)
#         vision_table.putNumberArray("X Coords 2", x_list)
#         vision_table.putNumberArray("Y Coords 2", y_list)
#         vision_table.putNumberArray("Z Coords 2", z_list)
#         vision_table.putNumberArray("Yaws 2", yaw_list)
#         vision_table.putNumberArray("Timestamps 2", timestamp_list)
#         # vision_table.putNumberArray("Z Euler Angles", z_euler_list)
#         # vision_table.putNumber("Best Tag ID", bestID)
#         # vision_table.putNumber("Best Tag X", bestX)
#         # vision_table.putNumber("Best Tag Y", bestY)
#         # vision_table.putNumber("Best Tag Z", bestZ)
#         # vision_table.putNumberArray("X Euler Angles", x_euler_list)
#         # vision_table.putNumberArray("Y Euler Angles", y_euler_list)

#         # if len(x_list) >= 2:
#         #     avg_x = sum(x_list) / len(x_list)
#         #     avg_y = sum(y_list) / len(y_list)

#         #     # Calculate distance to tag
#         #     distance = calcDistanceToTag(avg_x, avg_y)
#         #     vision_table.putNumber("DistanceToTag", distance)

# #detects apriltags for cam1
# def cam3TagDetect():
#         cam3_frame_time, cam3_input_img = cam3_input_stream.grabFrame(img)

#         if cam3_frame_time == 0:
#             output_stream.notifyError(cam3_input_stream.getError())

#         # Setting up tag info lists           
#         x_list = []
#         y_list = []
#         z_list = []
#         yaw_list = []
#         id_list = []
#         timestamp_list = []
#         # x_euler_list = []
#         # y_euler_list = []
#         # z_euler_list = []
#         # bestID = -1
#         # bestX = -1
#         # bestY = -1
#         # bestZ = -1

#         # Grayscale frame + detect
#         gray_img = cv2.cvtColor(cam3_input_img, cv2.COLOR_BGR2GRAY)
#         tag_info = detector.detect(gray_img)

#         # Filter out bad detections (low decision margin + out of bounds IDs)
#         filter_tags = [tag for tag in tag_info if tag.getDecisionMargin() > DETECTION_MARGIN_THRESHOLD]
#         filter_tags = [tag for tag in filter_tags if ((tag.getId() > 0) & (tag.getId() < 17))]
        
#         # Setting up best tag stuff
#         # if len(filter_tags) > 0:
#         #     bestTag = filter_tags[0]
#         #     for tag in filter_tags:
#         #         if tag.getDecisionMargin() > bestTag.getDecisionMargin():
#         #             bestTag = tag
#         #     bestID = bestTag.getId()
#         #     bestTagPos = tag_estimator.estimate(bestTag)
#         #     bestX = bestTagPos.getX()
#         #     bestY = bestTagPos.getY()
#         #     bestZ = bestTagPos.getZ()
#         #     vision_table.putNumber("Best Timestamp", ntcore._now())                    

#         # Send detections info over network tables
#         for tag in filter_tags:
#             tag_id = tag.getId()
#             # homography = tag.getHomographyMatrix()
#             # euler_list = rotationMatrixToEulerAngles(homography)
#             tag_pos = tag_estimator.estimate(tag)
#             x_list.insert(0, tag_pos.X())
#             y_list.insert(0, tag_pos.Y())
#             z_list.insert(0, tag_pos.Z())
#             yaw_list.insert(0, tag_pos.rotation().Z())
#             timestamp_list.insert(0, ntcore._now())
#             # x_list.insert(0, (center.y))
#             # y_list.insert(0, (center.x))
#             id_list.insert(0, tag_id)
#             # z_euler_list.insert(0, euler_list[2])
#             # x_euler_list.insert(euler_list[0] * 1000)
#             # y_euler_list.insert(euler_list[1] * 1000)

#             #pop lists in case they get too big to avoid memory issues
#             if len(x_list) > 10:
#                 x_list.pop()

#             if len(y_list) > 10:
#                 y_list.pop()
            
#             if len(z_list) > 10:
#                 z_list.pop()
            
#             if len(yaw_list) > 10:
#                 yaw_list.pop()

#             if len(id_list) > 10:
#                 id_list.pop()
                
#             if len(timestamp_list) > 10:
#                 timestamp_list.pop()
#             # if len(z_euler_list) > 10:
#             #     z_euler_list.pop()

#         vision_table.putNumberArray("IDs 3", id_list)
#         vision_table.putNumberArray("X Coords 3", x_list)
#         vision_table.putNumberArray("Y Coords 3", y_list)
#         vision_table.putNumberArray("Z Coords 3", z_list)
#         vision_table.putNumberArray("Yaws 3", yaw_list)
#         vision_table.putNumberArray("Timestamps 3", timestamp_list)
#         # vision_table.putNumberArray("Z Euler Angles", z_euler_list)
#         # vision_table.putNumber("Best Tag ID", bestID)
#         # vision_table.putNumber("Best Tag X", bestX)
#         # vision_table.putNumber("Best Tag Y", bestY)
#         # vision_table.putNumber("Best Tag Z", bestZ)
#         # vision_table.putNumberArray("X Euler Angles", x_euler_list)
#         # vision_table.putNumberArray("Y Euler Angles", y_euler_list)

#         # if len(x_list) >= 2:
#         #     avg_x = sum(x_list) / len(x_list)
#         #     avg_y = sum(y_list) / len(y_list)

#         #     # Calculate distance to tag
#         #     distance = calcDistanceToTag(avg_x, avg_y)
#         #     vision_table.putNumber("DistanceToTag", distance)

def main():
    while True:
        try:
            cam1TagDetect()
            # cam2TagDetect()
            # cam3TagDetect()
        except:
            continue

main()