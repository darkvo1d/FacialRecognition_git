import cv2
import os
import time


def run(vidPath, frames=360):
    start_time = time.time()
    # read video file
    photo_path = "/".join(vidPath.split("/")[0:2]) + "/images"
    res = os.path.isdir(photo_path)
    if not res:
        os.mkdir(photo_path)
    create_images(vidPath, photo_path, frames)
    stop_time = time.time()
    print("all convert time is %s" % (stop_time - start_time))
    photoNumber = len(os.listdir(photo_path))
    print("There are %s photos in total" % photoNumber)

def create_images(vidPath, photo_path, frames):
    vidPath = ".".join([vidPath.split(".")[0], "MP4"]) if not os.path.exists(vidPath) else vidPath
    vidcap = cv2.VideoCapture(vidPath)
    # judge whether it opens properly
    if vidcap.isOpened():
        totalFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        jump = int(totalFrames / frames)
        count = 1
        frame = jump
        # video frame count interval frequency
        while frame < totalFrames:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame)
            success, image = vidcap.read()
            # store as an image
            cv2.imwrite("{}/image".format(photo_path) + str(count) + ".jpg", image)
            count = count + 1
            frame += jump
            cv2.waitKey(1)
        vidcap.release()
    else:
        print("Video unable to process")