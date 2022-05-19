"""
import time
from src.encodings import Encodings
from src.recognize import Recognize
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to serialized db of all data")
args = vars(ap.parse_args())

studentPath = "data/sarvang" + args["input"]
studentName = "sarvang"

t1 = time.time()
encoding = Encodings(studentPath + "/learning")
encoding.encode()
t2 = time.time()
encodingsPath = studentPath + "/learning/" + studentName + "_encodings.pickle"
imagesPath = studentPath + "/images"
recognize = Recognize(encodingsPath, imagesPath)
result = recognize.recognize_images()
print(list(result))
t3 = time.time()

print(t2 - t1)
print(t3 - t2)
"""

import cv2
import time
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True, help="name of the video with extension avi")
args = vars(ap.parse_args())

current_milli_time = lambda: int(round(time.time() * 1000))
writer = None
vidPath = "dataset/learning_videos/" + args["name"]

# Camera feed
cap_cam = cv2.VideoCapture(1)
if not cap_cam.isOpened():
    print('Cannot open camera')
    exit()
ret, frame_cam = cap_cam.read()
if not ret:
    print('Cannot open camera stream')
    cap_cam.release()
    exit()

# Video feed
filename = 'data/sarvang/sarvang.mp4'
cap_vid = cv2.VideoCapture(filename)
if not cap_cam.isOpened():
    print('Cannot open video: ' + filename)
    cap_cam.release()
    exit()
ret, frame_vid = cap_vid.read()
if not ret:
    print('Cannot open video stream: ' + filename)
    cap_cam.release()
    cap_vid.release()
    exit()

# Specify maximum video time in milliseconds
max_time = 1000 * cap_vid.get(cv2.CAP_PROP_FRAME_COUNT) / cap_vid.get(cv2.CAP_PROP_FPS)

# Resize the camera frame to the size of the video
height = int(cap_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap_vid.get(cv2.CAP_PROP_FRAME_WIDTH))

# Starting from now, syncronize the videos
start = current_milli_time()

while True:
    # Capture the next frame from camera
    ret, frame_cam = cap_cam.read()
    if not ret:
        print('Cannot receive frame from camera')
        break
    frame_cam = cv2.resize(frame_cam, (width, height), interpolation=cv2.INTER_AREA)

    # Capture the frame at the current time point
    time_passed = current_milli_time() - start
    if time_passed > max_time:
        print('Video time exceeded. Quitting...')
        break
    ret = cap_vid.set(cv2.CAP_PROP_POS_MSEC, time_passed)
    if not ret:
        print('An error occured while setting video time')
        break
    ret, frame_vid = cap_vid.read()
    if not ret:
        print('Cannot read from video stream')
        break

    # Blend the two images and show the result
    tr = 0.3 # transparency between 0-1, show camera if 0
    frame = ((1-tr) * frame_cam.astype(np.float64) + tr * frame_vid.astype(np.float64)).astype(np.uint8)
    cv2.imshow('Transparent result', frame)

    if writer is None and vidPath is not None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(vidPath, fourcc, 20, (frame_cam.shape[1], frame_cam.shape[0]), True)
    # if the writer is not None, write the frame with recognized
    # faces to disk
    if writer is not None:
        writer.write(frame_cam)

    if cv2.waitKey(1) == 27: # ESC is pressed
        break

cap_cam.release()
cap_vid.release()
cv2.destroyAllWindows()