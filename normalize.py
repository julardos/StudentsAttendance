import numpy as np
import os
import cv2

class FaceDetector(object):
    def __init__(self, deploy_path, caffe_path):
        self.net = cv2.dnn.readNetFromCaffe(deploy_path, caffe_path)

    def __detect__(self, image):
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
            if confidence > 0.99:
                faces_coord.append((detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype("int"))
        faces_coord = np.array(faces_coord)
        return faces_coord

    def __cut_faces__(self, image):
        faces = []
        faces_coord = self.__detect__(image)
        for (startX, startY, endX, endY) in faces_coord:
            slicer = image[startY:endY, startX:endX]
            if (len(slicer)):
                faces.append(slicer)

        return faces

    def __resize__(self, images, size=(224, 224)):
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

    def normalize_faces(self, image):
        faces = self.__cut_faces__(image)
        faces = self.__resize__(faces, (150, 150))

        return faces