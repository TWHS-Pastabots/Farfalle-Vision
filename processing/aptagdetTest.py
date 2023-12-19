import math
import time
from cscore import CameraServer

import cv2
import json
import numpy as np
import robotpy_apriltag
from ntcore import NetworkTableInstance as nt

inst = nt.getDefault()
inst.startClient4("wpilibpi")
inst.setServerTeam(9418)
inst.startDSClient()
vision_table = inst.getTable('Vision')
time.sleep(0.5)

#nt.initialize(server = "10.94.18.2")

# Checks if a matrix is a valid rotation matrix.
#def isRotationMatrix(R) :
 #   Rt = np.transpose(R)
    # shouldBeIdentity = np.dot(Rt, R)
    # I = np.identity(3, dtype = R.dtype)
    # n = np.linalg.norm(I - shouldBeIdentity)
    # return n < 1e-6
 
# Calculates rotation matrix to euler angles
# of the euler angles ( x and z are swapped ).
# def rotationMatrixToEulerAngles(R) :
 
#     # assert(isRotationMatrix(R))
 
#     sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
 
#     singular = sy < 1e-6
 
#     if  not singular :
#         x = math.atan2(R[2,1] , R[2,2])
#         y = math.atan2(-R[2,0], sy)
#         z = math.atan2(R[1,0], R[0,0])
#     else :
#         x = math.atan2(-R[1,2], R[1,1])
#         y = math.atan2(-R[2,0], sy)
#         z = 0
 
#     return np.array([x, y, z])

def main():
    width = 480
    height = 360

    CameraServer.startAutomaticCapture()
    input_stream = CameraServer.getVideo()
    output_stream = CameraServer.putVideo('Processed', width, height)
    img = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    while True:
        frame_time, input_img = input_stream.grabFrame(img)

        id_list = []

        if frame_time == 0:
            output_stream.notifyError(input_stream.getError())
            continue

        detector = robotpy_apriltag.AprilTagDetector()
        detector.addFamily("tag16h5")

        DETECTION_MARGIN_THRESHOLD = 100

        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        tag_info = detector.detect(gray)

        filter_tags = [tag for tag in tag_info if tag.getDecisionMargin() > DETECTION_MARGIN_THRESHOLD]
        filter_tags = [tag for tag in filter_tags if ((tag.getId() > 0) & (tag.getId() < 9))]

        for tag in filter_tags:
            tag_id = tag.getId()
            id_list.append(tag_id)
           
        print(id_list)
        print("\n")
main()