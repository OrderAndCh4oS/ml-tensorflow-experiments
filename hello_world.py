import numpy as np
from keras import Sequential
from keras.layers import Dense


def hello_world():
    model = Sequential(Dense(units=1, input_shape=[1]))
    model.compile(optimizer='sgd', loss='mean_squared_error')

    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0])

    model.fit(xs, ys, epochs=500)

    return model.predict([10.0])


if __name__ == '__main__':
    print(hello_world())
