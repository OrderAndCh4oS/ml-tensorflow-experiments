import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Flatten

class StopTrainingWhenAccuracyReached(tf.keras.callbacks.Callback):
    accuracy: float

    def __init__(self, accuracy):
        self.accuracy = accuracy

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('accuracy') >= self.accuracy:  # Experiment with changing this value
            print(f"\nReached {self.accuracy} accuracy so cancelling training!")
            self.model.stop_training = True


class StopTrainingWhenLossLimitReached(tf.keras.callbacks.Callback):
    loss: float

    def __init__(self, loss):
        self.loss = loss

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('loss') >= self.loss:  # Experiment with changing this value
            print(f"\nReached {self.loss} loss so cancelling training!")
            self.model.stop_training = True


def fashion():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images = train_images / 255
    test_images = test_images / 255

    # index = 12
    #
    # np.set_printoptions(linewidth=320)
    #
    # print(f'LABEL: {train_labels[index]}')
    # print(f'{train_images[index]}')

    model = Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        Flatten(),
        Dense(28 * 28, activation=tf.nn.relu),
        Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callback = StopTrainingWhenAccuracyReached(0.9)
    model.fit(train_images, train_labels, epochs=10, callbacks=[callback])

    model.evaluate(test_images, test_labels)
    classifications = model.predict(test_images)
    max_index = np.where(classifications[0] == np.amax(classifications[0]))
    print((classifications[0][max_index[0][0]], max_index[0][0], test_labels[0]))


if __name__ == '__main__':
    fashion()
