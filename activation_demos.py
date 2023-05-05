import numpy as np
import tensorflow as tf


def softmax_demo(inputs):
    # Declare sample inputs and convert to a tensor
    inputs = tf.convert_to_tensor([inputs])
    print(f'input to softmax function: {inputs.numpy()}')

    # Feed the inputs to a softmax activation function
    outputs = tf.keras.activations.softmax(inputs)
    print(f'output of softmax function: {outputs.numpy()}')

    # Get the sum of all values after the softmax
    sum = tf.reduce_sum(outputs)
    print(f'sum of outputs: {sum}')

    # Get the index with highest value
    return np.argmax(outputs)


# def relu_demo(x):
#     if x > 0:
#         return x
#     return 0

if __name__ == '__main__':
    # print(relu_demo(1))
    prediction = softmax_demo([1.0, 3.0, 4.0, 2.0])
    print(f'class with highest probability: {prediction}')
