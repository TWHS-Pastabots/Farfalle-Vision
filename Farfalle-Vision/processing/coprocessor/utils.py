import constants
import numpy as np
from ntcore import NetworkTableInstance as nt
from cscore import CameraServer as cs
from cscore import VideoSource as vs
import cv2
import robotpy_apriltag as rptag
from scipy.spatial.transform import Rotation
import ntcore
import ultralytics
from ultralytics import YOLO

def launch_network_table():
    inst = nt.getDefault()
    inst.startClient4(constants.HOSTNAME)
    inst.setServerTeam(9418)
    inst.startDSClient()
    vision_table = inst.getTable(constants.VISION_TABLE)
    time.sleep(1.0)
    return vision_table

def init_yolo():
    ultralytics.check()
    model = YOLO(constants.YOLO_MODEL_PATH, task = "detect")
    time.sleep(1.0)
    return model

def launch_cam(name, cam_id):
    path = ""

    match cam_id:
        case 1:
            path = constants.USB1_PATH
        case 2:
            path = constants.USB2_PATH
        case 3:
            path = constants.USB3_PATH
        case 4:
            path = constants.USB4_PATH

    cam = cs.startAutonamaticCapture(name = name, path = path)
    cam.setConnectionStrategy(vs.ConnectionStrategy.kConnectionKeepOpen)
    cam.setResolution(constants.STREAM_WIDTH, constants.STREAM_HEIGHT)
    cam.setFPS(constants.FPS)

    cam_input_stream = cs.getVideo(camera = cam)

    return cam_input_stream

def get_output_stream():
    return cs.putVideo(name = constants.OUTPUT_STREAM_NAME, width = constants.STREAM_WIDTH,  height = constants.STREAM_HEIGHT)

def get_frame(cam_input_stream, output_stream):
    cam_frame_time, cam_input_img = cam_input_stream.grabFrame(constants.IMG_TEMPLATE)

    if cam1_frame_time == 0:  # If frame time is zero then there was no time between the last frame so no new data
        output_stream.notifyError(cam1_input_stream.getError())

    return cam_frame_time, cam_input_img

def get_tag_detector():
    detector = rptag.AprilTagDetector()
    detector.addFamily("tag36h11")
    return detector

def find_tags(img, detector, cam_id, vision_table):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tag_info = detector.detect(gray_img)

    # Filter out bad detections (low decision margin + out of bounds IDs)
    filter_tags = [tag for tag in tag_info if tag.getDecisionMargin() > DETECTION_MARGIN_THRESHOLD]
    filter_tags = [tag for tag in filter_tags if ((tag.getId() > 0) & (tag.getId() < 17))]

    # Process detections
    for tag in filter_tags:
        # Get corners of AprilTag
        corners = np.array(
            [[tag.getCorner(0).x, tag.getCorner(0).y],
            [tag.getCorner(1).x,  tag.getCorner(1).y],
            [tag.getCorner(2).x,  tag.getCorner(2).y],
            [tag.getCorner(3).x,  tag.getCorner(3).y]],
            dtype = np.float64
        )

        T_nt, R_nt = find_object_coords(corners, constants.TAG_POINTS)

        # Serialized tag information
        tag_serial_string = str(tag.getId()) + " " + str(T_nt[0]) + " " + str(T_nt[1]) + " " + str(T_nt[2]) + " "
        tag_serial_string += str(R_nt[0]) + " " + str(R_nt[1]) + " " + str(R_nt[2]) + " " + str(R_nt[3]) + " "
        tag_serial_string += str(ntcore._now())
        serialized_tags_list.insert(0, tag_serial_string)

        # Pop list in case it get too big to avoid memory issues
        if len(serialized_tags_list) > 10:
            serialized_tags_list.pop()

        # Send serialized tags over network tables
    vision_table.putStringArray("Serialized Tags, Cam" + cam_id, serialized_tags_list)

def find_gamepieces_cv(img, cam_id, vision_table, red):
    x_list = []
    y_list = []

    # Blur image and convert to HSV
    blurred_mat = cv2.GaussianBlur(img, (5, 5), 0)
    hsv_mat = cv2.cvtColor(blurred_mat, cv2.COLOR_BGR2HSV)

    # Isolate gamepiece color and get rid of holes/noise. Change dilation/erosion/morphology operations as desired
    color_segmented_mat = cv2.inRange(hsv_mat, constants.HSV_MIN, constants.HSV_MAX)

    # Merge two ranges together in case color is red
    if red:
        temp_mat = cv2.inRange(hsv_mat, constants.HSV_MIN2, constants.HSV_MAX2)
        color_segmented_mat = cv2.bitwise_or(color_segmented_mat, temp_mat)

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

    # TODO:: maybe do solvePnP with gamepieces as well

    vision_table.putNumberArray("CV Gamepiece X Coords, Cam" + cam_id, x_list)
    vision_table.putNumberArray("CV Gamepiece Y Coords, Cam" + cam_id, y_list)

def find_gamepieces_yolo(img, model, cam_id, vision_table):
    x_list = []
    y_list = []

    # Find rings
    results = model.predict(source = img, conf = constants.YOLO_CONF, save = False)

    # convert results to numpy array
    np_results = results[0]

    # Add ring detected centers to lists
    for i in range(len(np_results.boxes)):
        box = np_results.boxes.xyxy[i]
        x_list.append((box[0] + box[2]) / 2.0)
        y_list.append((box[1] + box[3]) / 2.0)

    vision_table.putNumberArray("YOLO Gamepiece X Coords, Cam" + cam_id, x_list)
    vision_table.putNumberArray("YOLO Gamepiece Y Coords, Cam" + cam_id, y_list)

def find_object_coords(object_corners, object_points):
    # Get tag space rotation vector and translation vector with solvePnP()
    _, r_vec, t_vec = cv2.solvePnP(
        objectPoints = object_points,
        imagePoints = object_corners,
        cameraMatrix = np.asarray(constants.INTRINSICS_MATRIX),
        distCoeffs = np.asarray(constants.DISTORTIONS),
        flags = cv2.SOLVEPNP_SQPNP
    )

    # Convert tag space to camera space
    r_mat = cv2.Rodrigues(r_vec)[0]

    # T = -r^T * t
    T_cs = -np.matrix(r_mat).T * np.matrix(t_vec)

    # Object translation vector
    T_nt = [
        float(T_cs[0]),
        float(T_cs[1]),
        float(T_cs[2])
    ]

    R_q = Rotation.from_matrix(r_mat).as_quat()

    # Object quaternion info
    R_nt = [
        float(R_q[0]),
        float(R_q[1]),
        float(R_q[2]),
        float(R_q[3]),
    ]

    # TODO on coprocessor:: convert coordinate systems (EDN - > NWU)
    # TODO on rio:: transform to robot space

    return T_nt, R_nt