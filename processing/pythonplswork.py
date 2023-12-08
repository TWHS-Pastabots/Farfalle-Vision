from cscore import CameraServer

import cv2
import json
import numpy
import robotpy_apriltag
import pynetworktables as nt

nt.initialize(server = "10.94.18.2")


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

            x_list.append((center.x - width / 2) / (width / 2))
            y_list.append((center.y - width / 2) / (width / 2))
            id_list.append(tag_id)

        nt.putNumberArray("IDs", id_list)
        print(id_list)
main()