import math
import time
import constants
import cv2
import json
import numpy as np
import robotpy_apriltag
from ntcore import NetworkTableInstance as nt
from cscore import CameraServer as cs

from processing.coprocessor.constants import MIN_CONTOUR_AREA

inst = nt.getDefault()
inst.startClient4("visiontwhs")
inst.setServerTeam(9418)
inst.startDSClient()
vision_table = inst.getTable("Vision")
time.sleep(2.0)

def main():
    width = 480
    height = 360

    usb1 = cs.startAutomaticCapture(name = "cam1", path = constants.USB1_PATH)
    cam1_input_stream = cs.getVideo(camera = usb1)
    cam1_output_stream = cs.putVideo(name = 'cam1', width = width, height = height)

    img = np.zeros(shape = (height, width, 3), dtype = np.uint8)

    while True:
        cam1_frame_time, cam1_input_img = cam1_input_stream.grabFrame(img)
        cv2.imshow("frame", cam1_input_img)

        x_list = []
        y_list = []

        if cam1_frame_time == 0:
            cam1_output_stream.notifyError(cam1_input_stream.getError())
            continue

        #Blur image and convert to HSV
        blurred_mat = cv2.GaussianBlur(cam1_input_img, (5, 5), 0)
        hsv_mat = cv2.cvtColor(blurred_mat, cv2.COLOR_BGR2HSV)

        # Isolate gamepiece color
        color_segmented_mat = cv2.inRange(hsv_mat, constants.HSV_MIN, constants.HSV_MAX)
        color_segmented_mat = cv2.dilate(color_segmented_mat, None, iterations = 2)
        color_segmented_mat = cv2.erode(color_segmented_mat, None, iterations = 2)

        # Find edges and contours
        contour_mat = cv2.Canny(color_segmented_mat, 50, 150)
        hierarchy, contours = cv2.findContours(contour_mat, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = [contour for contour in contours if cv2.contourArea(contour) > MIN_CONTOUR_AREA]

        # Find rotated rectangles for each contour
        gamepiece_rects = []
        for contour in contours:
            gamepiece_rects.append(cv2.minAreaRect(contour))

        for gamepiece_rect in gamepiece_rects:
            x_list.append(gamepiece_rect[0].center.x)
            y_list.append(gamepiece_rect[0].center.y)

        vision_table.putNumberArray("Gamepiece X Coords", x_list)
        vision_table.putNumberArray("Gamepiece Y Coords", y_list)

        if len(x_list) > 10:
            x_list.pop()

        if len(y_list) > 10:
            y_list.pop()

main()
