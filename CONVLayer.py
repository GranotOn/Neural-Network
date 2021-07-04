from Layer import Layer
from scipy import signal
import numpy as np


class CONVLayer(Layer):
    # inherit from base class Layer

    def __init__(self, input_shape, kernel_shape, layer_depth):
        self.input_shape = input_shape
        self.input_depth = input_shape[2]
        self.kernel_shape = kernel_shape
        self.layer_depth = layer_depth

        self.output_shape = (input_shape[0] - kernel_shape[0] + 1,
                             input_shape[1] - kernel_shape[1] + 1, layer_depth)
        self.weights = np.random.rand(kernel_shape[0], kernel_shape[1],
                                      self.input_depth, layer_depth) - 0.03
        self.bias = np.random.rand(layer_depth) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.zeros(self.output_shape)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                self.output[:, :, k] += signal.correlate2d(
                    self.input[:, :, d], self.weights[:, :, d, k],
                    'same   ') + self.bias[k]

        print(self.output)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        in_error = np.zeros(self.input_shape)
        delta_weights = np.zeros((self.kernel_shape[0], self.kernel_shape[1],
                                  self.input_depth, self.layer_depth))
        delta_bias = np.zeros(self.layer_depth)

        for k in range(self.layer_depth):
            for d in range(self.input_depth):
                in_error[:, :,
                         d] += signal.convolve2d(output_error[:, :, k],
                                                 self.weights[:, :, d,
                                                              k], 'full')
                delta_weights[:, :, d,
                              k] = signal.correlate2d(self.input[:, :, d],
                                                      output_error[:, :,
                                                                   k], 'valid')
            delta_bias[k] = self.layer_depth * np.sum(output_error[:, :, k])

        self.weights -= learning_rate * delta_weights
        self.bias -= learning_rate * delta_bias
        return in_error