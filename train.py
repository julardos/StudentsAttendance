import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import cv2
import numpy as np

class TrainData :
    def __init__(self, datasets):
        base_dir = datasets
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=20,
            horizontal_flip=True,
            shear_range=0.2,
            validation_split=0.4
        )

        self.train_generator = train_datagen.flow_from_directory(
                base_dir,
                target_size=(150, 150),
                batch_size=4,
                class_mode='categorical',
                subset="training")

        self.validation_generator = train_datagen.flow_from_directory(
                base_dir,
                target_size=(150, 150),
                batch_size=4,
                class_mode='categorical',
                subset="validation")

    def train_data(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3,3), activation='relu',
                                   input_shape=(150, 150, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(26, activation='softmax')
        ])

        self.model.compile(loss='categorical_crossentropy',
                      optimizer=tf.optimizers.Adam(),
                      metrics=['accuracy'])

        self.history = self.model.fit(
              self.train_generator,
              steps_per_epoch=30,
              epochs=20,
              validation_data=self.validation_generator,
              validation_steps=3,
              verbose=2)

        return self.train_plot()

    def train_plot(self):
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

    def predict(self, img, faces_list):
        img = cv2.cvtColor(cv2.resize(img, (150, 150)), cv2.COLOR_RGB2BGR)
        x = np.expand_dims(img, axis=0)
        image = np.vstack([x])
        classes = self.model.predict(image, batch_size=4)
        i = np.argmax(classes)
        print(classes, i, faces_list[i])

        return faces_list[i]