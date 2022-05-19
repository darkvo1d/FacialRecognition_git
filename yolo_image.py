import cv2
from pytorchyolo import detect, models

#get class list
classList = open("yolo-coco/coco.names", "r")
classList = classList.readlines()
classList = list(map(lambda x: x.replace("\n",""), classList))
#print(classList)

# Load the YOLO model
model = models.load_model("yolo-coco/yolov4.cfg", "yolo-coco/yolov4.weights")

# Load the image as a numpy array
img = cv2.imread("data\sarvang2\images\image135.jpg")

# Convert OpenCV bgr to rgb
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Runs the YOLO model on the image
detected_classes = [boxes[-1] for boxes in detect.detect_image(model, img)]

print("Detected_classes : ", ", ".join([classList[int(x)] for x in detected_classes]))
# Output will be a numpy array in the following format:
# [[x1, y1, x2, y2, confidence, class]]