# This is a sample Python script.

import os
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import shutil
import urllib.request
import urllib.request
import zipfile

import tensorflow as tf


# Downloads the pictures used in the CNN
def human_or_horse_images():
    url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
    file_name = "horse-or-human.zip"
    training_dir = 'horse-or-human/training/'
    urllib.request.urlretrieve(url, file_name)
    zip_ref = zipfile.ZipFile(file_name, 'r')
    zip_ref.extractall(training_dir)
    zip_ref.close()

    # File name
    file = 'horse-or-human.zip'
    # File location
    location = "/home/comlab/Projects/MLForCoders/Chapter3/"
    # Path
    path = os.path.join(location, file)
    # Remove the file 'horse-or-human.zip'
    os.remove(path)

    validation_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip"
    validation_file_name = "validation-horse-or-human.zip"
    validation_dir = 'horse-or-human/validation/'
    urllib.request.urlretrieve(validation_url, validation_file_name)
    zip_ref = zipfile.ZipFile(validation_file_name, 'r')
    zip_ref.extractall(validation_dir)
    zip_ref.close()

    # File name
    file = 'validation-horse-or-human.zip'
    # File location
    location = "/home/comlab/Projects/MLForCoders/Chapter3/"
    # Path
    path = os.path.join(location, file)
    # Remove the file 'horse-or-human.zip'
    os.remove(path)


def delete_images():
    location = "/home/comlab/Projects/MLForCoders/Chapter3/"
    directory = "horse-or-human"
    path = os.path.join(location, directory)
    # removing directory
    shutil.rmtree(path)


# Implementing Convolutional Neural Networks
def examplePage37():
    data = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = data.load_data()
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                               input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(training_images, training_labels, epochs=10)
    model.evaluate(test_images, test_labels)
    classifications = model.predict(test_images)
    print(classifications[0])
    print(test_labels[0])


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    human_or_horse_images()

    examplePage37()

    delete_images()
