import argparse
from src.recognize import Recognize
import pickle
from main import process_data as pd
from pytorchyolo import models

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--student", required=True, help="path to a students data")
ap.add_argument("-i", "--image", required=False, help="image name with extension")

args = vars(ap.parse_args())

__DATA = "data/"
__RESULT_STRINGS = dict({"unknown_face": "Unknown faces detected in ", "multiple_faces": "Multiple faces detected in ", "no_face": "No faces detected in"})
results = []
model = models.load_model("yolo-coco/yolov4.cfg", "yolo-coco/yolov4.weights")

if not args["image"]:
    results.append(pd(args["student"]))
    for result in results:
        for key in result.keys():
            print("{} : {}".format(__RESULT_STRINGS[key], list(result[key])))
        print("\n\n")
else:
    encodingsPath = __DATA + args["student"] + "/learning/" + args["student"] + "_encodings.pickle"
    data = pickle.loads(open(encodingsPath, "rb").read())
    imagePath = "data/" + args["student"] + "/images/" + args["image"]
    Recognize.recognize_image(data, imagePath, model)
