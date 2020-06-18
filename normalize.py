import numpy as np
import os
import cv2

class FaceDetector(object):
    def __init__(self, deploy_path, caffe_path):
        self.net = cv2.dnn.readNetFromCaffe(deploy_path, caffe_path)

    def detect(self, image, biggest_only=True):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        (h, w) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(
            image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        faces_coord = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                faces_coord.append((detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int"))
        faces_coord = np.array(faces_coord)
        return faces_coord

    def cut_faces(self, image, faces_coord):
        faces = []

        for (startX, startY, endX, endY) in faces_coord:
            slicer = image[startY:endY, startX:endX]
            if (len(slicer)):
                faces.append(slicer)

        return faces

    def resize(self, images, size=(224, 224)):
        images_norm = []
        for image in images:
            if image.shape < size:
                image_norm = cv2.resize(image, size,
                                        interpolation=cv2.INTER_AREA)
            else:
                image_norm = cv2.resize(image, size,
                                        interpolation=cv2.INTER_CUBIC)
            images_norm.append(image_norm)

        return images_norm

    def normalize_faces(self, image, faces_coord):
        faces = self.cut_faces(image, faces_coord)
        faces = self.resize(faces)

        return faces