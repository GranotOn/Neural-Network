from Layer import Layer
import numpy as np


class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()

    def forward_propagation(self, inputs):
        self.inputs = inputs
        inputs -= np.max(inputs)
        exponents = np.exp(inputs)
        exponents_sum = np.sum(exponents)
        self.outputs = exponents / exponents_sum
        return self.outputs

    def backward_propagation(self, output_gradient, learning_rate=0.02):
        """
        :param output_gradient:
        :return:
        """
        # dL/dx = dy/dx * dL/dy
        softmax_gradient = np.diag(self.outputs) - np.outer(
            self.outputs, self.outputs)
        input_gradient = np.dot(output_gradient, softmax_gradient)
        return input_gradient
