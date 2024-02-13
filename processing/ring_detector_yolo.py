import time
from cscore import CameraServer as cs
from cscore import MjpegServer as MJServer
from cscore import VideoSource as vs
from cscore import CvSink as sink
from cscore import CvSource as source
from cscore import VideoMode
import cv2
import tensorflow as tf
import numpy as np
import ultralytics
from ultralytics import YOLO

#check everything is fine with ultralytics for YOLO
#ultralytics.checks()

#load model path
modelPath = "/home/vision/Documents/Code/best.pt"

def main():
    #frame dimensions
    height = 480
    width = 640

    #set up camera with dev path
    usb2 = cs.startAutomaticCapture(name = "cam2", path = "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.2:1.0-video-index0")
    usb2.setConnectionStrategy(vs.ConnectionStrategy.kConnectionKeepOpen)
    
    #mjpeg server for raw stream display
    mj = MJServer("cam2_stream", 1185)
    mj.setSource(usb2)

    #setting up second mjpeg server sink -> source -> mjepg server
    cam2_sink = sink("cam2_sink")
    cam2_sink.setSource(usb2)

    cam2_source = source("cam2_source", pixelFormat = VideoMode.PixelFormat.kMJPEG, width = width, height = height, fps  = 20)
    mj2 = MJServer("cam2_processed", 1186)
    mj2.setSource(cam2_source)

    #wait a bit
    time.sleep(2)

    #setting up some input stream stuff
    cam2_input_stream = cs.getVideo(camera = usb2)
    img = np.zeros(shape = (height, width, 3), dtype = np.uint8)
    
    #remove scientific notation
    np.set_printoptions(suppress = True)

    #load yolo model
    model = YOLO(modelPath, task = "detect")

    while True:
        #get frame from stream and send to models
        cam2_frame_time, cam2_input_img = cam2_sink.grabFrame(img)
        #cv2.imshow("Frame", cam2_input_img)
        
        ring_img = cam2_input_img
        #ring_img = np.asarray(cam2_input_img, dtype = np.float32)
        #cv2.imshow("Frame", ring_img)
        #cv2.waitKey(1)
        results = model.predict(source = ring_img, conf = 0.25, save = False)

        #convert results to numpy array
        np_results = results[0]
        #.numpy()
        #print(np_results)
        
        #mark ring contours and post to source of mj2
        if len(np_results) == 0:
            print("No Ring")
        else:
            for i in range(len(np_results.boxes)):
                box = np_results.boxes.xyxy[i]
                #print(box)
                #print(box[0])
                #image, (topleft), (bottomright), color, thickness px
                cv2.rectangle(ring_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 3)
        
        cv2.imshow("Rings", ring_img)
        cv2.waitKey(1)
        cam2_source.putFrame(ring_img)
        
main()