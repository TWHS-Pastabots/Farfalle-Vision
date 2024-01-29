import math
import time
#from typing import Self
from cscore import CameraServer as cs
from cscore import VideoSource as vs
from cscore import MjpegServer as MJServer
from cscore import VideoMode as vm

import cv2
import json
import numpy as np
import robotpy_apriltag
from ntcore import NetworkTableInstance as nt


inst = nt.getDefault()
inst.startClient4("visiontwhs")
inst.setServerTeam(9418)
inst.startDSClient()
vision_table = inst.getTable('Vision')
time.sleep(2.0)


def main():
    width = 480
    height = 360

    #cs.startAutomaticCapture()
    usb2 = cs.startAutomaticCapture(name = "cam2", path = "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.2:1.0-video-index0")
    #usb2.setConnectionStrategy(vs.ConnectionStrategy.kConnectionKeepOpen)

    #To view stream type hostname + port into browser? right?
    mj = MJServer("cam2_stream", 1185)
    mj.setSource(usb2)

    cam2_input_stream = cs.getVideo(camera = usb2)
    cam2_output_stream = cs.putVideo(name = 'cam2', width = width, height = height)

    img = np.zeros(shape=(height, width, 3), dtype=np.uint8)

    while True:
        cam2_frame_time, cam2_input_img = cam2_input_stream.grabFrame(img)
        cv2.imshow("frame", cam2_input_img)
        cv2.waitKey(1)
        
        if cam2_frame_time == 0:
            cam2_output_stream.notifyError(cam2_input_stream.getError())
            continue

        detector = robotpy_apriltag.AprilTagDetector()
        detector.addFamily("tag16h5")

        DETECTION_MARGIN_THRESHOLD = 100

        gray2 = cv2.cvtColor(cam2_input_img, cv2.COLOR_BGR2GRAY)
        tag_info2 = detector.detect(gray2)
        
        filter_tags2 = [
            tag for tag in tag_info2 if tag.getDecisionMargin() > DETECTION_MARGIN_THRESHOLD]
        filter_tags2 = [tag for tag in filter_tags2 if (
            (tag.getId() > 0) & (tag.getId() < 9))]
        
        if len(filter_tags2) > 0:
            print("Cam2 detects!\n")

main()
