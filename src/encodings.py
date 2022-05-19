from imutils import paths
import face_recognition
import pickle
import cv2
import os


class Encodings:
    def __init__(self, datasetPath):
        self.datasetPath = datasetPath
        self.detectionMethod = "cnn"
        # initialize the list of known encodings and known names
        self.knownEncodings = []
        self.knownNames = []
        self.imagePaths = list(paths.list_images(self.datasetPath))

    def encode(self):
        # grab the paths to the input images in our dataset
        print("[INFO] quantifying faces...")

        # loop over the image paths
        for (i, imagePath) in enumerate(self.imagePaths):
            # extract the person name from the image path
            print("[INFO] processing image {}/{}".format(i + 1, len(self.imagePaths)))
            name = imagePath.split(os.path.sep)[-2]
            # load the input image and convert it from BGR (OpenCV ordering)
            # to dlib ordering (RGB)
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # detect the (x, y)-coordinates of the bounding boxes
            # corresponding to each face in the input image
            boxes = face_recognition.face_locations(rgb, model=self.detectionMethod)
            # compute the facial embedding for the face
            encodings = face_recognition.face_encodings(rgb, boxes)
            # loop over the encodings
            for encoding in encodings:
                # add each encoding + name to our set of known names and
                # encodings
                self.knownEncodings.append(encoding)
                self.knownNames.append(name)
        # dump the facial encodings + names to disk
        print("[INFO] serializing encodings...")
        data = {"encodings": self.knownEncodings, "names": self.knownNames}
        f = open(self.datasetPath + "/" + self.datasetPath.split("/")[1] + "_encodings.pickle", "wb")
        f.write(pickle.dumps(data))
        f.close()
