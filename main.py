import numpy as np
import os
import cv2
from normalize import FaceDetector
from train import TrainData
import uuid
import sys

labels_dic = {}

def prepare_dataset(base_dir, directory) :
    images = []
    labels = []
    global labels_dic
    people = [person for person in os.listdir(base_dir)]
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir(base_dir + person):
            images.append(cv2.imread(base_dir + person + '/' + image, 0))
            labels.append(person)

    labels = np.array(labels)


    for label in range(len(labels_dic)):
        path = os.path.join(directory, labels_dic[label])
        try:
            os.mkdir(path)
        except FileExistsError:
            print("Created Directory ", path)

    detector = FaceDetector("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")
    for (j, image) in enumerate(images):
        faces_coord = detector.detect(image, True)
        faces = detector.normalize_faces(image, faces_coord)
        for i, face in enumerate(faces):
            cv2.imwrite(directory + '/' + labels[j] + "/" + str(uuid.uuid4()) + '.jpeg', faces[i])

def train_dataset(base_dir) :
    return TrainData(base_dir).train_data()

def predict(model, image) :
    model.predict()

def main(argv):
    base_dir = 'datasets'
    directory = 'new_datasets'
    image = "datasets/2103181015/2103181015.jpeg"
    for action in argv:
        if action == "prepare_dataset" :
            prepare_dataset(base_dir, directory)
        elif action == "train":
            model = train_dataset(directory)
        elif action == "predict":
            try:
                predict(model, )
            except NameError:
                train_dataset(directory)


if __name__ == '__main__':
    main(sys.argv[1:])