import cv2
import time
import traceback
from PIL import Image

bp = r"P:\Python Projects\FacialRecognition\ImageBasic"
intervals = 30  # sleeping for 30 seconds in between image captures


def cord2s(cord):
    sLr = [-10, 2560]
    sRr = [2560, 4000]
    if len(cord) > 4:
        cord = cord[4]  # can accept getwinoplacement too
    x = (cord[0] + cord[2]) / 2.0
    # print("center:",x)
    gr = lambda sir: (x - sir[0]) / (sir[1] - sir[0])
    # gr=lambda sir:(gr_(sir)-0.2)/(0.8-0.2)
    if sLr[0] <= x <= sLr[1]:
        return "L", gr(sRr)
    if sRr[0] <= x <= sRr[1]:
        return "R", gr(sRr)
    return -1, -1


def captureImage(self, name):
    print("Hello")
    while True:
        cam = cv2.VideoCapture(0)
        print("STARTING")
        imgName = lambda _i=0: "{bp}/{personName}_{tm}.png".format(personName=name, tm=time.time(), bp=bp)
        try:
            while True:
                iName = imgName()
                Image.fromarray(cam.read()[1]).save(iName, "PNG")
                time.sleep(intervals)
                print(iName)
        except AttributeError:
            print("cam stuck:\n" + traceback.format_exc())
            time.sleep(5)
            print("restarting..\n")
        cam.release()
