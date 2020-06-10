import numpy as np
import os
import cv2

base_dir = 'datasets'
images = []
labels = []
labels_dic = {}
people = [person for person in os.listdir(base_dir)]
for i, person in enumerate(people):
    labels_dic[i] = person
    for image in os.listdir(base_dir + person):
        images.append(cv2.imread(base_dir + person + '/' + image, 0))
        labels.append(person)

labels = np.array(labels)


class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)

    def detect(self, image, biggest_only=True):
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30, 30)
        biggest_only = True
        faces_coord = self.classifier.detectMultiScale(image,
                                                       scaleFactor=scale_factor,
                                                       minNeighbors=min_neighbors,
                                                       minSize=min_size,
                                                       flags=cv2.CASCADE_SCALE_IMAGE)
        return faces_coord


def cut_faces(image, faces_coord):
    faces = []

    for (x, y, w, h) in faces_coord:
        w_rm = int(0.3 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])

    return faces


def resize(images, size=(224, 224)):
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


def normalize_faces(image, faces_coord):
    faces = cut_faces(image, faces_coord)
    faces = resize(faces)

    return faces


count = 0
directory = "new_datasets"
if not os.path.exists('new_datasets'):
    os.makedirs('new_datasets')

for label in range(len(labels_dic)):
    path = os.path.join(directory, labels_dic[label])
    os.mkdir(labels_dic[label])

for (i, image) in enumerate(images):
    detector = FaceDetector("/content/haarcascade_frontalface_default.xml")
    faces_coord = detector.detect(image, True)
    faces = normalize_faces(image, faces_coord)
    os.chdir(directory + "/" + labels[i])
    for i, face in enumerate(faces):
        cv2.imwrite('%s.jpeg' % (count), faces[i])
        count += 1