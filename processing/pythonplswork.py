import math
import time
import threading
import numpy as np
from cscore import CameraServer as cs
from cscore import VideoSource as vs
from cscore import CvSink as sink
import cv2
#from ultralytics import YOLO
import robotpy_apriltag as rptag
from wpimath.geometry import Transform3d, Translation3d, Rotation3d
from ntcore import NetworkTableInstance as nt
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
usb1 = cs.startAutomaticCapture(name = "cam1", path = "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.1:1.0-video-index0")
usb1.setConnectionStrategy(vs.ConnectionStrategy.kConnectionKeepOpen)
usb1.setResolution(width, height)
usb1.setFPS(fps)

usb2 = cs.startAutomaticCapture(name = "cam2", path = "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.2:1.0-video-index0")
usb2.setConnectionStrategy(vs.ConnectionStrategy.kConnectionKeepOpen)
usb1.setResolution(width, height)
usb2.setFPS(fps)

# Insert dev path later
usb3 = cs.startAutomaticCapture(name = "cam3", path = "")
usb3.setConnectionStrategy(vs.ConnectionStrategy.kConnectionKeepOpen)
usb3.setResolution(width, height)
usb3.setFPS(fps)

cam1_input_stream = cs.getVideo(camera = usb1)
cam2_input_stream = cs.getVideo(camera = usb2)
cam3_input_stream = cs.getVideo(camera = usb3)

# Setting up sink for cam2
cam2_sink = sink("cam2_sink")
cam2_sink.setSource(usb2)

time.sleep(0.5)

output_stream = cs.putVideo(name = 'Cam Stream', width = width,  height = height)
img = np.zeros(shape = (height, width, 3), dtype = np.uint8)

# Remove scientific notation
np.set_printoptions(suppress = True)

# Set up tag detectors + estimators
detector = rptag.AprilTagDetector()
detector.addFamily("tag36h11")

detector2 = rptag.AprilTagDetector()
detector2.addFamily("tag36h11")

detector3 = rptag.AprilTagDetector()
detector3.addFamily("tag36h11")

DETECTION_MARGIN_THRESHOLD = 90

# Tagsize (m), fx, fy, cx, cy
tag_estimator_conf = rptag.AprilTagPoseEstimator.Config(0.2159, 699.3778103158814, 677.7161226393544, 345.6059345433618, 207.12741326228522)
tag_estimator = rptag.AprilTagPoseEstimator(tag_estimator_conf)
tag_estimator2 = rptag.AprilTagPoseEstimator(tag_estimator_conf)
tag_estimator3 = rptag.AprilTagPoseEstimator(tag_estimator_conf)

# load yolo model
#modelPath = "/home/vision/Documents/Code/best.pt"
#model = YOLO(modelPath, task = "detect")

time.sleep(0.5)

# Checks if a matrix is a valid rotation matrix.
# def isRotationMatrix(R):
#     Rt = np.transpose(R)
#     shouldBeIdentity = np.dot(Rt, R)
#     I = np.identity(3, dtype = R.dtype)
#     n = np.linalg.norm(I - shouldBeIdentity)
#     return n < 1e-6

# # Calculates rotation matrix to euler angles of the euler angles ( x and z are swapped ).
# def rotationMatrixToEulerAngles(R):
#     #assert(isRotationMatrix(R))
#     sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
#     singular = sy < 1e-6

#     if not singular:
#         x = math.atan2(R[2, 1], R[2, 2])
#         y = math.atan2(-R[2, 0], sy)
#         z = math.atan2(R[1, 0], R[0, 0])
#     else:
#         x = math.atan2(-R[1, 2], R[1, 1])
#         y = math.atan2(-R[2, 0], sy)
#         z = 0

#     return np.array([x, y, z])


# Tag_size_meters = 0.1
# Focal_Length_Pixels = 600
# FOV_horizontal_degrees = 60  # Horizontal field of view of the camera in degrees
# FOV_vertical_degrees = 40    # Vertical field of view of the camera in degrees

# def calcDistanceToTag(x_tag_pixels, y_tag_pixels):
#     # Calculate the angle of the AprilTag from the center of the image
#     angle_x = (x_tag_pixels - width / 2) * (FOV_horizontal_degrees / width)
#     angle_y = (y_tag_pixels - height / 2) * (FOV_vertical_degrees / height)

#     # Calculate the distance to the AprilTag using trigonometry (assuming small angles)
#     distance_x = Tag_size_meters / (2 * math.tan(math.radians(angle_x / 2)))
#     distance_y = Tag_size_meters / (2 * math.tan(math.radians(angle_y / 2)))

#     # Average the distances from both horizontal and vertical angles
#     distance = (distance_x + distance_y) / 2

#      #hoping this works and ideally should return the distance to the AprilTag based on the camera
#     return distance


# Detects apriltags for cam1
def cam1TagDetect():
    while True:
        cam1_frame_time, cam1_input_img = cam1_input_stream.grabFrame(img)

        if cam1_frame_time == 0:
            output_stream.notifyError(cam1_input_stream.getError())

        # Setting up tag info lists           
        x_list = []
        y_list = []
        z_list = []
        yaw_list = []
        id_list = []
        timestamp_list = []
        # x_euler_list = []
        # y_euler_list = []
        # z_euler_list = []
        # bestID = -1
        # bestX = -1
        # bestY = -1
        # bestZ = -1

        # Grayscale frame + detect
        gray_img = cv2.cvtColor(cam1_input_img, cv2.COLOR_BGR2GRAY)
        tag_info = detector.detect(gray_img)

        # Filter out bad detections (low decision margin + out of bounds IDs)
        filter_tags = [tag for tag in tag_info if tag.getDecisionMargin() > DETECTION_MARGIN_THRESHOLD]
        filter_tags = [tag for tag in filter_tags if ((tag.getId() > 0) & (tag.getId() < 17))]
        
        # Setting up best tag stuff
        # if len(filter_tags) > 0:
        #     bestTag = filter_tags[0]
        #     for tag in filter_tags:
        #         if tag.getDecisionMargin() > bestTag.getDecisionMargin():
        #             bestTag = tag
        #     bestID = bestTag.getId()
        #     bestTagPos = tag_estimator.estimate(bestTag)
        #     bestX = bestTagPos.getX()
        #     bestY = bestTagPos.getY()
        #     bestZ = bestTagPos.getZ()
        #     vision_table.putNumber("Best Timestamp", ntcore._now())                    

        # Send detections info over network tables
        for tag in filter_tags:
            tag_id = tag.getId()
            # homography = tag.getHomographyMatrix()
            # euler_list = rotationMatrixToEulerAngles(homography)
            tag_pos = tag_estimator.estimate(tag)
            x_list.insert(0, tag_pos.getX())
            y_list.insert(0, tag_pos.getY())
            z_list.insert(0, tag_pos.getZ())
            yaw_list.insert(0, tag_pos.getRotation().getZ())
            timestamp_list.insert(0, ntcore._now())
            # x_list.insert(0, (center.y))
            # y_list.insert(0, (center.x))
            id_list.insert(0, tag_id)
            # z_euler_list.insert(0, euler_list[2])
            # x_euler_list.insert(euler_list[0] * 1000)
            # y_euler_list.insert(euler_list[1] * 1000)

            #pop lists in case they get too big to avoid memory issues
            if len(x_list) > 10:
                x_list.pop()

            if len(y_list) > 10:
                y_list.pop()
            
            if len(z_list) > 10:
                z_list.pop()
            
            if len(yaw_list) > 10:
                yaw_list.pop()

            if len(id_list) > 10:
                id_list.pop()
            if len(timestamp_list) > 10:
                timestamp_list.pop()
            # if len(z_euler_list) > 10:
            #     z_euler_list.pop()

        vision_table.putNumberArray("IDs", id_list)
        vision_table.putNumberArray("X Coords", x_list)
        vision_table.putNumberArray("Y Coords", y_list)
        vision_table.putNumberArray("Z Coords", z_list)
        vision_table.putNumberArray("Yaws", yaw_list)
        vision_table.putNumberArray("Timestamps", timestamp_list)
        # vision_table.putNumberArray("Z Euler Angles", z_euler_list)
        # vision_table.putNumber("Best Tag ID", bestID)
        # vision_table.putNumber("Best Tag X", bestX)
        # vision_table.putNumber("Best Tag Y", bestY)
        # vision_table.putNumber("Best Tag Z", bestZ)
        # vision_table.putNumberArray("X Euler Angles", x_euler_list)
        # vision_table.putNumberArray("Y Euler Angles", y_euler_list)

        # if len(x_list) >= 2:
        #     avg_x = sum(x_list) / len(x_list)
        #     avg_y = sum(y_list) / len(y_list)

        #     # Calculate distance to tag
        #     distance = calcDistanceToTag(avg_x, avg_y)
        #     vision_table.putNumber("DistanceToTag", distance)

# def cam2RingDetect():
#     while True:
#         cam2_frame_time, cam2_input_img = cam2_sink.grabFrame(img)

#         if cam2_frame_time == 0:
#             output_stream.notifyError(cam2_input_stream.getError())

#         #get model resultsSS
#         results = model.predict(source = cam2_input_img, conf = 0.25, save = False)
#         np_results = results[0]

#         ring_center_x = []
#         ring_center_y = []
            
#         #mark ring contours
#         if len(np_results) == 0:
#             print("No Ring")
#         else:
#             for i in range(len(np_results.boxes)):
#                 box = np_results.boxes.xyxy[i]
#                 l = int(box[0])
#                 t = int(box[1])
#                 b = int(box[2])
#                 r = int(box[3])
    
#                 #image, (topleft), (bottomright), color, thickness px
#                 #cv2.rectangle(cam2_input_img, (l, t), (b, r), (0, 255, 0), 3)
#                 ring_center_x.append((l + r) / 2.0)
#                 ring_center_y.append((b + t) / 2.0)
            
#         vision_table.putNumberArray("Ring Center X Coords", ring_center_x)
#         vision_table.putNumberArray("Ring Center Y Coords", ring_center_y)

#         if len(ring_center_x) > 5:
#             ring_center_x.pop()

#         if len(ring_center_y) > 5:
#             ring_center_y.pop()
            
#             #cam2_source.putFrame(cam2_input_img)
            
#             #cv2.imshow("Rings", cam2_input_img)
#             #cv2.waitKey(1)

def cam2TagDetect():
    while True:
        cam2_frame_time, cam2_input_img = cam2_input_stream.grabFrame(img)

        if cam2_frame_time == 0:
            output_stream.notifyError(cam2_input_stream.getError())

        # Setting up tag info lists           
        x_list = []
        y_list = []
        z_list = []
        yaw_list = []
        id_list = []
        timestamp_list = []
        # x_euler_list = []
        # y_euler_list = []
        # z_euler_list = []
        # bestID = -1
        # bestX = -1
        # bestY = -1
        # bestZ = -1

        # Grayscale frame + detect
        gray_img = cv2.cvtColor(cam2_input_img, cv2.COLOR_BGR2GRAY)
        tag_info = detector.detect(gray_img)

        # Filter out bad detections (low decision margin + out of bounds IDs)
        filter_tags = [tag for tag in tag_info if tag.getDecisionMargin() > DETECTION_MARGIN_THRESHOLD]
        filter_tags = [tag for tag in filter_tags if ((tag.getId() > 0) & (tag.getId() < 17))]
        
        # Setting up best tag stuff
        # if len(filter_tags) > 0:
        #     bestTag = filter_tags[0]
        #     for tag in filter_tags:
        #         if tag.getDecisionMargin() > bestTag.getDecisionMargin():
        #             bestTag = tag
        #     bestID = bestTag.getId()
        #     bestTagPos = tag_estimator.estimate(bestTag)
        #     bestX = bestTagPos.getX()
        #     bestY = bestTagPos.getY()
        #     bestZ = bestTagPos.getZ()
        #     vision_table.putNumber("Best Timestamp", ntcore._now())                    

        # Send detections info over network tables
        for tag in filter_tags:
            tag_id = tag.getId()
            # homography = tag.getHomographyMatrix()
            # euler_list = rotationMatrixToEulerAngles(homography)
            tag_pos = tag_estimator.estimate(tag)
            x_list.insert(0, tag_pos.getX())
            y_list.insert(0, tag_pos.getY())
            z_list.insert(0, tag_pos.getZ())
            yaw_list.insert(0, tag_pos.getRotation().getZ())
            timestamp_list.insert(0, ntcore._now())
            # x_list.insert(0, (center.y))
            # y_list.insert(0, (center.x))
            id_list.insert(0, tag_id)
            # z_euler_list.insert(0, euler_list[2])
            # x_euler_list.insert(euler_list[0] * 1000)
            # y_euler_list.insert(euler_list[1] * 1000)

            #pop lists in case they get too big to avoid memory issues
            if len(x_list) > 10:
                x_list.pop()

            if len(y_list) > 10:
                y_list.pop()
            
            if len(z_list) > 10:
                z_list.pop()
            
            if len(yaw_list) > 10:
                yaw_list.pop()

            if len(id_list) > 10:
                id_list.pop()

            if len(timestamp_list) > 10:
                timestamp_list.pop()
            # if len(z_euler_list) > 10:
            #     z_euler_list.pop()

        vision_table.putNumberArray("IDs 2", id_list)
        vision_table.putNumberArray("X Coords 2", x_list)
        vision_table.putNumberArray("Y Coords 2", y_list)
        vision_table.putNumberArray("Z Coords 2", z_list)
        vision_table.putNumberArray("Yaws 2", yaw_list)
        vision_table.putNumberArray("Timestamps 2", timestamp_list)
        # vision_table.putNumberArray("Z Euler Angles", z_euler_list)
        # vision_table.putNumber("Best Tag ID", bestID)
        # vision_table.putNumber("Best Tag X", bestX)
        # vision_table.putNumber("Best Tag Y", bestY)
        # vision_table.putNumber("Best Tag Z", bestZ)
        # vision_table.putNumberArray("X Euler Angles", x_euler_list)
        # vision_table.putNumberArray("Y Euler Angles", y_euler_list)

        # if len(x_list) >= 2:
        #     avg_x = sum(x_list) / len(x_list)
        #     avg_y = sum(y_list) / len(y_list)

        #     # Calculate distance to tag
        #     distance = calcDistanceToTag(avg_x, avg_y)
        #     vision_table.putNumber("DistanceToTag", distance)

#detects apriltags for cam1
def cam3TagDetect():
    while True:
        cam3_frame_time, cam3_input_img = cam3_input_stream.grabFrame(img)

        if cam3_frame_time == 0:
            output_stream.notifyError(cam3_input_stream.getError())

        # Setting up tag info lists           
        x_list = []
        y_list = []
        z_list = []
        yaw_list = []
        id_list = []
        timestamp_list = []
        # x_euler_list = []
        # y_euler_list = []
        # z_euler_list = []
        # bestID = -1
        # bestX = -1
        # bestY = -1
        # bestZ = -1

        # Grayscale frame + detect
        gray_img = cv2.cvtColor(cam3_input_img, cv2.COLOR_BGR2GRAY)
        tag_info = detector.detect(gray_img)

        # Filter out bad detections (low decision margin + out of bounds IDs)
        filter_tags = [tag for tag in tag_info if tag.getDecisionMargin() > DETECTION_MARGIN_THRESHOLD]
        filter_tags = [tag for tag in filter_tags if ((tag.getId() > 0) & (tag.getId() < 17))]
        
        # Setting up best tag stuff
        # if len(filter_tags) > 0:
        #     bestTag = filter_tags[0]
        #     for tag in filter_tags:
        #         if tag.getDecisionMargin() > bestTag.getDecisionMargin():
        #             bestTag = tag
        #     bestID = bestTag.getId()
        #     bestTagPos = tag_estimator.estimate(bestTag)
        #     bestX = bestTagPos.getX()
        #     bestY = bestTagPos.getY()
        #     bestZ = bestTagPos.getZ()
        #     vision_table.putNumber("Best Timestamp", ntcore._now())                    

        # Send detections info over network tables
        for tag in filter_tags:
            tag_id = tag.getId()
            # homography = tag.getHomographyMatrix()
            # euler_list = rotationMatrixToEulerAngles(homography)
            tag_pos = tag_estimator.estimate(tag)
            x_list.insert(0, tag_pos.getX())
            y_list.insert(0, tag_pos.getY())
            z_list.insert(0, tag_pos.getZ())
            yaw_list.insert(0, tag_pos.getRotation().getZ())
            timestamp_list.insert(0, ntcore._now())
            # x_list.insert(0, (center.y))
            # y_list.insert(0, (center.x))
            id_list.insert(0, tag_id)
            # z_euler_list.insert(0, euler_list[2])
            # x_euler_list.insert(euler_list[0] * 1000)
            # y_euler_list.insert(euler_list[1] * 1000)

            #pop lists in case they get too big to avoid memory issues
            if len(x_list) > 10:
                x_list.pop()

            if len(y_list) > 10:
                y_list.pop()
            
            if len(z_list) > 10:
                z_list.pop()
            
            if len(yaw_list) > 10:
                yaw_list.pop()

            if len(id_list) > 10:
                id_list.pop()
                
            if len(timestamp_list) > 10:
                timestamp_list.pop()
            # if len(z_euler_list) > 10:
            #     z_euler_list.pop()

        vision_table.putNumberArray("IDs 3", id_list)
        vision_table.putNumberArray("X Coords 3", x_list)
        vision_table.putNumberArray("Y Coords 3", y_list)
        vision_table.putNumberArray("Z Coords 3", z_list)
        vision_table.putNumberArray("Yaws 3", yaw_list)
        vision_table.putNumberArray("Timestamps 3", timestamp_list)
        # vision_table.putNumberArray("Z Euler Angles", z_euler_list)
        # vision_table.putNumber("Best Tag ID", bestID)
        # vision_table.putNumber("Best Tag X", bestX)
        # vision_table.putNumber("Best Tag Y", bestY)
        # vision_table.putNumber("Best Tag Z", bestZ)
        # vision_table.putNumberArray("X Euler Angles", x_euler_list)
        # vision_table.putNumberArray("Y Euler Angles", y_euler_list)

        # if len(x_list) >= 2:
        #     avg_x = sum(x_list) / len(x_list)
        #     avg_y = sum(y_list) / len(y_list)

        #     # Calculate distance to tag
        #     distance = calcDistanceToTag(avg_x, avg_y)
        #     vision_table.putNumber("DistanceToTag", distance)

def main():
        #set up threads and run
        t1 = threading.Thread(target = cam1TagDetect, name = "cam1 thread")
        t2 = threading.Thread(target = cam2TagDetect, name = "cam2 thread")
        t3 = threading.Thread(target = cam3TagDetect, name = "cam3 thread")

        t1.start()
        t2.start()      
        t3.start()

main()
