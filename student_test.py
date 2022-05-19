import argparse
import os
import timeit
from src.encodings import Encodings
from src.recognize import Recognize
import src.get_images as gi
from pytorchyolo import models

# ap = argparse.ArgumentParser()
# ap.add_argument("-d", "--data", required=True, help="path to serialized db of all data")
# args = vars(ap.parse_args())

__DATA = "data"
__RESULT_STRINGS = dict({"unknown_face": "Unknown faces detected in ", "multiple_faces": "Multiple faces detected in ", "no_face": "No faces detected in", "cell_phone": "Cell phone detected in"})
model = models.load_model("yolo-coco/yolov4.cfg", "yolo-coco/yolov4.weights")

def process_data(studentName, model):
    # set the student path
    studentPath = __DATA + "/" + studentName

    # extract learning images from learning video
    learningImagePath = studentPath + "/learning"
    learningVidPath = learningImagePath + "/" + studentName + "_learning.mp4"
    print("Extracting images from learning video")
    gi.create_images(learningVidPath, learningImagePath, 10)

    # learn from the images
    print("Learning Images")
    encoding = Encodings(learningImagePath)
    encoding.encode()
    encodingsPath = learningImagePath + "/" + studentName + "_encodings.pickle"

    # identify the fishy images
    vidPath = studentPath + "/" + studentName + ".mp4"
    print("Extracting Images from test video")
    gi.run(vidPath)
    imagesPath = studentPath + "/images"
    print("Recognizing Images")
    recognize = Recognize(encodingsPath, imagesPath, model)
    return recognize.recognize_images()


students = os.listdir(__DATA)
students = ['sarvang2']
results = []

start = timeit.default_timer()
print(start)
for student in students:
    results.append(process_data(student, model))
print(results)
for result in results:
    for key in result.keys():
        print("{} : {}".format(__RESULT_STRINGS[key], list(result[key])))
    print("\n\n")
end = timeit.default_timer()
print(end-start)