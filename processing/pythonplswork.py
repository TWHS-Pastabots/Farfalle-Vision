import threading
from cscore import CameraServer

import cv2
import json
import numpy
import robotpy_apriltag
from ntcore import NetworkTables as nt

cond = threading.Condition()
notified = [False]

def connectionListener(connected, info):
    print(info, '; Connected=%s' % connected)
    with cond:
        notified[0] = True
        cond.notify()

nt.initialize(server = "10.94.18.2")
nt.addConnectionListener(connectionListener, immediateNotify=True)

with cond:
    print("Waiting")
    if not notified[0]:
        cond.wait()

def main():
    width = 480
    height = 360

    CameraServer.startAutomaticCapture()
    input_stream = CameraServer.getVideo()
    output_stream = CameraServer.putVideo('Processed', width, height)
    img = numpy.zeros(shape=(height, width, 3), dtype=numpy.uint8)

    while True:
        frame_time, input_img = input_stream.grabFrame(img)

        x_list = []
        y_list = []
        id_list = []
        homography_list = []

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
            center = tag.getCenter()
            homography = tag.getHomographyMatrix()

            x_list.append((center.x - width / 2) / (width / 2))
            y_list.append((center.y - width / 2) / (width / 2))
            id_list.append(tag_id)
            homography_list.append(homography)

        nt.putNumberArray("IDs", id_list)
        nt.putNumberArray("X Coords", x_list)
        nt.putNumberArray("Y Coords", y_list)
        nt.putNumberArray("Homography for euler angles", homography_list)
        #print(id_list)
        print(homography_list)
main()