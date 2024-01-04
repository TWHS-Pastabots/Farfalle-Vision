import math
import time
#from typing import Self
from cscore import CameraServer

import cv2
import json
import numpy as np
import robotpy_apriltag
import ntcore
from ntcore import NetworkTableInstance as nt


inst = nt.getDefault()
inst.startClient4("wpilibpi")
inst.setServerTeam(9418)
inst.startDSClient()
vision_table = inst.getTable('Vision')
# visionTopic = inst.getIntegerArrayTopic("/Vision/Tags")
# visionTopic = vision_table.getIntegerArrayTopic("Tags")
# gtopic = inst.getIntegerArrayTopic("Tags")
# visionTopic = ntcore.IntegerArrayTopic(gtopic)
time.sleep(2.0)

# def __init__(self, visionTopic: ntcore.IntegerArrayTopic):
#     self.intPub = visionTopic.publish()

# __init__(visionTopic)
# nt.initialize(server = "10.94.18.2")

# Checks if a matrix is a valid rotation matrix.


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# of the euler angles ( x and z are swapped ).


def rotationMatrixToEulerAngles(R):

    # assert(isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

# def close(self):
    # self.intPub.close()


def main():
    width = 480
    height = 360

    CameraServer.startAutomaticCapture()
    input_stream = CameraServer.getVideo()
    output_stream = CameraServer.putVideo('Processed', width, height)
    img = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    while True:
        #Self.intPub.setDefault(0)

        frame_time, input_img = input_stream.grabFrame(img)

        x_list = []
        y_list = []
        id_list = []
        x_euler_list = []
        y_euler_list = []
        z_euler_list = []

        if frame_time == 0:
            output_stream.notifyError(input_stream.getError())
            continue

        detector = robotpy_apriltag.AprilTagDetector()
        detector.addFamily("tag16h5")

        DETECTION_MARGIN_THRESHOLD = 100

        gray = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        tag_info = detector.detect(gray)

        filter_tags = [
            tag for tag in tag_info if tag.getDecisionMargin() > DETECTION_MARGIN_THRESHOLD]
        filter_tags = [tag for tag in filter_tags if (
            (tag.getId() > 0) & (tag.getId() < 9))]

        for tag in filter_tags:
            tag_id = tag.getId()
            # self.intPub.set(tag_id, 0)
            center = tag.getCenter()
            homography = tag.getHomographyMatrix()
            euler_list = rotationMatrixToEulerAngles(homography)

            x_list.insert(0, (center.y - width / 2) / (width / 2) * 1000)
            y_list.insert(0, (center.x - width / 2) / (width / 2) * 1000)
            id_list.insert(0, tag_id * 1000)
            z_euler_list.insert(0, euler_list[2] * 1000)
           # x_euler_list.append(euler_list[0])
           # y_euler_list.append(euler_list[1])

        vision_table.putNumberArray("IDs", id_list)
        vision_table.putNumberArray("X Coords", x_list)
        vision_table.putNumberArray("Y Coords", y_list)
        vision_table.putNumberArray("Z Euler Angles", z_euler_list)
        # vision_table.putNumberArray("X Euler Angles", x_euler_list)
        # vision_table.putNumberArray("Y Euler Angles", y_euler_list)

        if len(x_list) > 10:
           x_list.pop()

        if len(y_list) > 10:
           y_list.pop()

        if len(id_list) > 5:
           x_list.pop()

        if len(z_euler_list) > 10:
           x_list.pop()
        

        #print(x_euler_list)
        #print(y_euler_list)
        #print(z_euler_list)
        print(id_list)
        print(x_list)
        print(y_list)
        print(z_euler_list)
        print("\n")


main()
# close()
