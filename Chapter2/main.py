# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def examplePage18():
    l0 = Dense(units=1, input_shape=[1])
    model = Sequential([l0])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)
    model.fit(xs, ys, epochs=500)
    print(model.predict([10.0]))
    print("Here is what I learned: {}".format(l0.get_weights()))

#Stopping the training when a certain number of generations (epochs) is reached
def examplePage26():
    data = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = data.load_data()
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(training_images, training_labels, epochs=5)


    classifications = model.predict(test_images)
    print(classifications[0])  #Prints the probability for the test item to match a label
    print(test_labels[0])      #Tells the true label of the item

#Stopping the training when a certain accuracy is reached
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.95):
            print("\nReached 95% accuracy so cancelling training!")
            self.model.stop_training = True

def examplePage31():
    callbacks = myCallback()

    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    training_images = training_images / 255.0
    test_images = test_images / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
#    examplePage18()
#    examplePage26()
    examplePage31()

