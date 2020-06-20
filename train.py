import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from normalize import FaceDetector
import uuid
import pickle

class TrainData :
    def __init__(self, datasets, temp_directory, deploy_path, caffe_path):
        self.base_dir = datasets
        self.directory = temp_directory
        self.detector = FaceDetector(deploy_path, caffe_path)


    def train_data_flow(self):
        label = self.prepare_dataset()
        self.dataset_parsing()
        self.fitting_data()
        # self.fitting_plot()


    def prepare_dataset(self):
        images = []
        labels = []
        labels_dic = {}
        people = [person for person in os.listdir(self.base_dir)]
        for i, person in enumerate(people):
            labels_dic[i] = person
            for image in os.listdir(self.base_dir + '/' + person):
                images.append(cv2.imread(self.base_dir + '/' + person + '/' + image, 0))
                labels.append(person)

        labels = np.array(labels)

        for label in range(len(labels_dic)):
            path = os.path.join(self.directory, labels_dic[label])
            try:
                os.mkdir(path)
            except FileExistsError:
                print("Created Directory ", path)

        for (j, image) in enumerate(images):
            faces = self.detector.normalize_faces(image)
            for i, face in enumerate(faces):
                cv2.imwrite(self.directory + '/' + labels[j] + "/" + str(uuid.uuid4()) +
                            '.jpeg', faces[i])

        a_file = open("temp/dataset.pkl", "wb")
        pickle.dump(labels_dic, a_file)
        a_file.close()

    def dataset_parsing(self):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            horizontal_flip=True,
            shear_range=0.2,
            validation_split=0.4
        )

        self.train_generator = train_datagen.flow_from_directory(
            self.base_dir,
            target_size=(150, 150),
            batch_size=4,
            class_mode='categorical',
            subset="training")

        self.validation_generator = train_datagen.flow_from_directory(
            self.base_dir,
            target_size=(150, 150),
            batch_size=4,
            class_mode='categorical',
            subset="validation")

    def create_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                                   input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(26, activation='softmax')
        ])

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.optimizers.Adam(),
                      metrics=['accuracy'])
        return model


    def fitting_data(self):
        model = self.create_model()
        self.history = model.fit(
              self.train_generator,
              steps_per_epoch=15,
              epochs=20,
              validation_data=self.validation_generator,
              validation_steps=3,
              verbose=2)

        model.summary()
        model.save_weights('temp/weight.h5')

    def fitting_plot(self):
        plt.figure(figsize=(20,10))
        plt.subplot(1, 2, 1)
        plt.ylabel('Loss', fontsize=16)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.legend(loc='upper right')

        plt.subplot(1, 2, 2)
        plt.ylabel('Accuracy', fontsize=16)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.show()

    def load(self):
        model = self.create_model()
        model.load_weights('temp/weight.h5')
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.optimizers.Adam(),
                      metrics=['accuracy'])
        a_file = open("temp/dataset.pkl", "rb")
        faces_list = pickle.load(a_file)
        a_file.close()

        return (model, faces_list)

    def predict(self, img):
        model, faces_list = self.load()
        img = cv2.cvtColor(cv2.resize(img, (150, 150)), cv2.COLOR_RGB2BGR)
        x = np.expand_dims(img, axis=0)
        image = np.vstack([x])
        classes = model.predict(image, batch_size=4)
        i = np.argmax(classes)
        return faces_list[i]