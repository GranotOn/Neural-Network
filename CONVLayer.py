from Layer import Layer
from scipy import signal
import numpy as np


class CONVLayer(Layer):
    def __init__(self,
                 input_size,
                 filter_size,
                 out_feature_maps,
                 stride=1,
                 same_padding=True):
        self.inputs = None
        self.outputs = None
        # Setting up all the parameters
        self.stride = stride
        self.same_padding = same_padding
        # input_depth is number of channels in the Input
        self.input_depth, self.input_height, self.input_width = input_size
        self.filter_height, self.filter_width = filter_size
        self.output_depth = out_feature_maps
        self.output_height, self.output_width = self.input_height, self.input_width

        self.modified_inputs = None
        self.filter_weights = np.random.standard_normal(
            (out_feature_maps, self.input_depth, self.filter_height,
             self.filter_width))

        if not same_padding:
            self.output_height, self.output_width = (self.input_height - self.filter_height) // stride + 1, \
                                                    (self.input_width - self.filter_width) // stride + 1
        self.biases = np.random.randn(out_feature_maps, self.output_height,
                                      self.output_width)

    def _add_padding(self, feature_maps):
        padding_height = self.filter_height // 2
        padding_width = self.filter_width // 2
        depth, height, width = feature_maps.shape
        # Assuming stride of size 1
        modified_inputs = np.zeros((depth, height + self.filter_height - 1,
                                    width + self.filter_width - 1))
        modified_inputs[:, padding_height:-padding_height,
                        padding_width:-padding_width] = feature_maps
        return modified_inputs

    def _convolve_forward(self, inputs, filter_weights):
        outputs = np.zeros(
            (self.output_depth, self.output_height, self.output_width))
        for out_feature_map_index in range(self.output_depth):
            for row in range(self.output_height):
                for col in range(self.output_width):
                    target_input = inputs[:, row:(row + self.filter_height),
                                          col:(col + self.filter_width)]
                    outputs[out_feature_map_index][row][col] = \
                        np.sum(filter_weights[out_feature_map_index] * target_input)
        return outputs

    def forward_propagation(self, inputs):
        self.inputs = inputs
        self.modified_inputs = inputs
        if self.same_padding:
            self.modified_inputs = self._add_padding(self.inputs)
        # self.outputs = self.biases + self._convolve_forward(self.modified_inputs, self.filter_weights)
        # return self.outputs
        return self._convolve_forward(self.modified_inputs,
                                      self.filter_weights) + self.biases

    def _weights_derivatives(self, modified_inputs, output_gradient):
        """
        :param modified_inputs: Cached modified(padded) inputs for back propagation
        :param output_gradient: dL/dY
        :return: dL/dW
        """
        weight_derivatives = np.zeros((self.output_depth, self.input_depth,
                                       self.filter_height, self.filter_width))
        for out_index in range(self.output_depth):
            for in_index in range(self.input_depth):
                for row in range(self.filter_height):
                    for col in range(self.filter_width):
                        target_input = \
                            modified_inputs[in_index, row:(row + self.input_height), col:(col + self.input_width)]
                        weight_derivatives[out_index][in_index][row][col] = \
                            np.sum(target_input * output_gradient[out_index])
        return weight_derivatives

    def _convolve_backward(self, output_gradient, filter_weights):
        # Flip the filter weights
        filter_weights = filter_weights[:, :, ::-1, ::-1]
        results = np.zeros(
            (self.input_depth, self.input_height, self.input_width))
        for input_feature_map_index in range(self.input_depth):
            for row in range(self.input_height):
                for col in range(self.input_width):
                    target_output = output_gradient[:,
                                                    row:(row +
                                                         self.filter_height),
                                                    col:(col +
                                                         self.filter_width)]
                    results[input_feature_map_index][row][col] = \
                        np.sum(filter_weights[:, input_feature_map_index] * target_output)
        return results

    def backward_propagation(self, output_gradient, learning_rate=0.9):
        modified_gradient = self._add_padding(output_gradient)
        # dL/dX Input Gradient
        input_gradient = self._convolve_backward(modified_gradient,
                                                 self.filter_weights)
        self.biases -= learning_rate * output_gradient
        self.filter_weights -= learning_rate * self._weights_derivatives(
            self.modified_inputs, output_gradient)
        return input_gradient