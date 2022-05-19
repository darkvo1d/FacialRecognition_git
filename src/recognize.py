# import the necessary packages
import face_recognition
import pickle
import cv2
from imutils import paths
from PIL import Image
import os
from pytorchyolo import detect


class Recognize:
    def __init__(self, encodingsPath, imagesPath, yoloModel):
        self.tolerance = 0.55
        self.detectionMethod = "cnn"
        self.result = []
        self.encodingsPath = encodingsPath
        self.data = pickle.loads(open(encodingsPath, "rb").read())
        self.imagesPath = imagesPath
        self.yoloModel = yoloModel

    def check_image(self, imagePath):
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes corresponding
        # to each face in the input image, then compute the facial embeddings
        # for each face
        # print("[INFO] recognizing faces...")
        boxes = face_recognition.face_locations(rgb, model=self.detectionMethod)
        encodings = face_recognition.face_encodings(rgb, boxes)

        detected_classes = self.getObjects(self.yoloModel, imagePath)
        flag = 67 in detected_classes
        if flag:
            self.extract_problem_image(imagePath, 3, False)

        # no face is found
        if len(encodings) > 1:
            self.extract_problem_image(imagePath, 2, boxes)
            return imagePath.split("/")[-1], 2, flag
        elif len(encodings) == 0:
            self.extract_problem_image(imagePath, 0, False)
            return imagePath.split("/")[-1], 0, flag

        # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known encodings
            matches = face_recognition.compare_faces(self.data["encodings"], encoding, self.tolerance)
            # check to see if we have found a match
            if True in matches:
                if flag:
                    return imagePath.split("/")[-1], 3, flag
                return False
        cv2.waitKey(1)
        # unknown face is found
        self.extract_problem_image(imagePath, 1, boxes)
        return imagePath.split("/")[-1], 1, flag

    @staticmethod
    def format_results(results):
        print(results)
        formattedResults = dict({"unknown_face": [], "no_face": [], "multiple_faces": [], "cell_phone": []})
        for result in results:
            if result and result[2]:
                formattedResults["cell_phone"].append(result[0].split("\\")[-1])
            if result[1] == 0:
                formattedResults["no_face"].append(result[0].split("\\")[-1])
            elif result[1] == 1:
                formattedResults["unknown_face"].append(result[0].split("\\")[-1])
            elif result[1] == 2:
                formattedResults["multiple_faces"].append(result[0].split("\\")[-1])

        return formattedResults

    @staticmethod
    def extract_problem_image(imagePath, problemType, boxes):
        problemImage = cv2.imread(imagePath)
        rgb_image = cv2.cvtColor(problemImage, cv2.COLOR_BGR2RGB)
        sizeFactor = problemImage.shape[1] / float(rgb_image.shape[1])
        if boxes:
            for (top, right, bottom, left) in boxes:
                # rescale the face coordinates
                top = int(top * sizeFactor)
                right = int(right * sizeFactor)
                bottom = int(bottom * sizeFactor)
                left = int(left * sizeFactor)
                cv2.rectangle(problemImage, (left, top), (right, bottom), (0, 255, 0), 2)

        directory = imagePath.split("images")[0] + "problemImages/"
        res = os.path.isdir(directory)
        if not res:
            os.mkdir(directory)

        if problemType == 0:
            folderName = "no_face"
        elif problemType == 1:
            folderName = "unknown_face"
        elif problemType == 2:
            folderName = "multiple_faces"
        elif problemType == 3:
            folderName = "cell_phone"

        folderDirectory = directory + folderName + "/"
        res = os.path.isdir(folderDirectory)
        if not res:
            os.mkdir(folderDirectory)
        Image.fromarray(problemImage).save(folderDirectory + imagePath.split("images")[1][1:-4] + ".png", "PNG")

    def recognize_images(self):
        images = list(paths.list_images(self.imagesPath))
        results = filter(lambda x: x, map(lambda image: self.check_image(image), images))
        formattedResult = self.format_results(list(results))
        return formattedResult

    @staticmethod
    def recognize_image(data, imagePath):
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        print("[INFO] recognizing faces...")
        boxes = face_recognition.face_locations(rgb, model="cnn")
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # loop over the facial embeddings
        for encoding in encodings:
            matches = face_recognition.compare_faces(data["encodings"], encoding, 0.55)
            name = "Unknown"
            if True in matches:
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                for i in matchedIdxs:
                    name = imagePath.split("/")[1]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
            names.append(name)

        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.imshow("Image", image)
        cv2.waitKey(0)

    @staticmethod
    def getObjects(yoloModel, imgPath):
        img = cv2.imread(imgPath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = detect.detect_image(yoloModel, img)  # Output will be a numpy array in the following format: [[x1, y1, x2, y2, confidence, class]]
        detected_classes = [boxes[-1] for boxes in detect.detect_image(yoloModel, img)]
        return detected_classes