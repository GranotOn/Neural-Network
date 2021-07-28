import numpy as np


# activation function and its derivative
def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1 - np.tanh(x)**2


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size


def relu(m):
    m[m < 0] = 0
    return m


def relu_prime(x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return x


def sigmoid(inputs):
    return 1 / (1 + np.exp(-1 * inputs, dtype='float64'))


def sigmoid_prime(outputs):
    return outputs * (1 - outputs)


def cross_entropy(prediction, true_prediction):
    return -1 * np.sum(true_prediction * np.log(prediction))


def cross_entropy_prime(prediction, true_prediction):
    # return (prediction - true_prediction) / len(prediction) # Wrong but working, Why?
    return -1 * true_prediction / prediction