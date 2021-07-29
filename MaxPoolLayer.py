from Layer import Layer
import numpy as np


class MaxPoolLayer(Layer):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.input_gradient_indices = None

    def forward_propagation(self, inputs):
        self.inputs = inputs
        depth, height, width = inputs.shape
        self.outputs = np.zeros(
            (inputs.shape[0], inputs.shape[1] // 2, inputs.shape[2] // 2))
        # Input Gradient Indices will be used in the Backward Pass
        self.input_gradient_indices = np.zeros_like(self.inputs)
        for depth_index in range(depth):
            for row in range(0, height, 2):
                for col in range(0, width, 2):
                    self.outputs[depth_index][row // 2][col // 2] = np.max(
                        inputs[depth_index, row:row + 2, col:col + 2]) / 255
                    # Figure out the row index and the column index of the Max Element in the 2x2 Grid
                    max_index = np.argmax(inputs[depth_index, row:row + 2,
                                                 col:col + 2])
                    row_index = max_index // 2
                    col_index = max_index % 2
                    self.input_gradient_indices[depth_index][row + row_index][
                        col + col_index] = 1
        return self.outputs

    def backward_propagation(self, output_gradient, learning_rate=0.2):
        depth, height, width = self.inputs.shape
        input_gradient = np.zeros_like(self.inputs)
        for depth_index in range(depth):
            for row in range(0, height, 2):
                for col in range(0, width, 2):
                    input_gradient[depth_index, row:row + 2, col:col +
                                   2] = output_gradient[depth_index][row //
                                                                     2][col //
                                                                        2]
        input_gradient = input_gradient * self.input_gradient_indices
        return input_gradient
