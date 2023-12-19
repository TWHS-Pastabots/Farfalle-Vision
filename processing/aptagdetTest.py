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
# inst.startDSClient()
vision_table = inst.getTable('Vision')
time.sleep(0.5)

# cap = cv2.VideoCapture(0)


def main():
    width = 480
    height = 360

    CameraServer.startAutomaticCapture()
    input_stream = CameraServer.getVideo()
    output_stream = CameraServer.putVideo('Processed', width, height)
    img = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    while True:
        #     # Capture frame-by-frame
        # ret, frame = cap.read()
        # # if frame is read correctly ret is True
        # if not ret:
        #     print("Can't receive frame (stream end?). Exiting ...")
        #     break
        # # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # # Display the resulting frame
        # cv2.imshow('frame', gray)
        # if cv2.waitKey(1) == ord('q'):
        #     break
        start_time = time.time
        frame_time, input_img = input_stream.grabFrame(img)

        id_list = []
        centerx_list = []
        centery_list = []

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
            center = tag.getCenter()

            id_list.append(tag_id)
            centerx_list.append(center.x)
            centery_list.append(center.y)
        vision_table.putNumberArray("Tag Ids", id_list)

        print(centerx_list)
        print(centery_list)
        print(id_list)


main()
# cap.release()
# cv2.destroyAllWindows()
