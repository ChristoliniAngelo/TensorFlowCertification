# =======================================================================================================
# PROBLEM C3
#
# Build a CNN based classifier for Cats vs Dogs dataset.
# Your input layer should accept 150x150 with 3 bytes color as the input shape.
# This is unlabeled data, use ImageDataGenerator to automatically label it.
# Don't use lambda layers in your model.
#
# The dataset used in this problem is originally published in https://www.kaggle.com/c/dogs-vs-cats/data
#
# Desired accuracy and validation_accuracy > 72%
# ========================================================================================================

import tensorflow as tf
import urllib.request
import zipfile
import tensorflow as tf
import os
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.72 and logs.get('val_accuracy')>0.72):
            print("\nDesired accuracy and validation_accuracy has been reached, Finish.")
            self.model.stop_training = True
def solution_C3():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/cats_and_dogs.zip'
    urllib.request.urlretrieve(data_url, 'cats_and_dogs.zip')
    local_file = 'cats_and_dogs.zip'
    zip_ref = zipfile.ZipFile(local_file, 'r')
    zip_ref.extractall('data/')
    zip_ref.close()

    BASE_DIR = 'data/cats_and_dogs_filtered'
    train_dir = os.path.join(BASE_DIR, 'train')
    validation_dir = os.path.join(BASE_DIR, 'validation')

    training_datagen = ImageDataGenerator(
        rescale=1 / 255,
        rotation_range=40,
        horizontal_flip=True,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # YOUR IMAGE SIZE SHOULD BE 150x150
    # Make sure you used "binary"
    train_generator = training_datagen.flow_from_directory(train_dir,
                                                           class_mode='binary',
                                                           batch_size=64,
                                                           target_size=(150, 150))

    validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                            class_mode='binary',
                                                            batch_size=32,
                                                            target_size=(150, 150))

    callback = myCallback()
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                  optimizer=RMSprop(learning_rate=0.001),
                  metrics=['accuracy'])

    model.fit(
        train_generator,
        epochs=30,
        validation_data=validation_generator,
        callbacks=callback)

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_C3()
    model.save("model_C3.h5")
