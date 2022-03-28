# This is a sample Python script.

import os
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import shutil
import urllib.request
import urllib.request
import zipfile

import tensorflow as tf
from keras.applications.densenet import layers
from keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


# Google colab of the example here:
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Horse_or_Human_NoValidation.ipynb
def example_human_horse_chapter3():
    training_dir = 'horse-or-human/training/'
    validation_dir = 'horse-or-human/validation/'

    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1 / 255)
    train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=(300, 300),
        class_mode='binary'
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                               input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['accuracy'])
    history = model.fit(
        train_generator,
        epochs=15
    )

    validation_datagen = ImageDataGenerator(rescale=1 / 255)
    validation_generator = train_datagen.flow_from_directory(
        validation_dir,
        target_size=(300, 300),
        class_mode='binary'
    )
    history = model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator
    )
    model.summary()


def example_image_augmentation_chapter3():
    training_dir = 'horse-or-human/training/'
    validation_dir = 'horse-or-human/validation/'

    # All images will be rescaled by 1./255 and added the augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=(300, 300),
        class_mode='binary'
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                               input_shape=(300, 300, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['accuracy'])
    history = model.fit(
        train_generator,
        epochs=15
    )

    validation_datagen = ImageDataGenerator(rescale=1 / 255)
    validation_generator = train_datagen.flow_from_directory(
        validation_dir,
        target_size=(300, 300),
        class_mode='binary'
    )
    history = model.fit(
        train_generator,
        epochs=15,
        validation_data=validation_generator
    )
    model.summary()


#Google colab (better to see it there):
#https://colab.research.google.com/github/lmoroney/tfbook/blob/master/chapter3/transfer_learning-cats-dogs.ipynb
def example_transfer_learning_chapter3():
    #Downloads the weights of the pretrained model and runs it with said weights
    weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
    weights_file = "inception_v3.h5"
    urllib.request.urlretrieve(weights_url, weights_file)
    pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                    include_top=False,
                                    weights=None)
    pre_trained_model.load_weights(weights_file)

    #Freezes the pretrained model and choses the mixed7 output
    for layer in pre_trained_model.layers:
        layer.trainable = False
    last_layer = pre_trained_model.get_layer('mixed7')
    print('last layer output shape: ', last_layer.output_shape)
    last_output = last_layer.output

    #Adds dense layers to the model
    #Flatten the output layer to 1 dimension
    x = layers.Flatten()(last_output)
    #Add a fully connected layer with 1, 024 hidden units and ReLU activation
    x = layers.Dense(1024, activation='relu')(x)
    #Add a final sigmoid layer for classification
    x = layers.Dense(1, activation='sigmoid')(x)

    model = Model(pre_trained_model.input, x)
    model.compile(optimizer=RMSprop(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    # Delete weights_file
    file = 'inception_v3.h5'
    location = "/home/comlab/Projects/MLForCoders/Chapter3/"
    path = os.path.join(location, file)
    os.remove(path)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    human_or_horse_images()
#    example_human_horse_chapter3()
#    examplePage37()
    example_transfer_learning_chapter3()
    delete_images()
