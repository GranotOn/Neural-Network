class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # Compute the output of Y to input X
    def forward_propogation(self, input):
        raise NotImplementedError

    # Computes dE / dX for a given dE / dY
    def backward_propogation(self, output_error, learning_rate):
        raise NotImplementedError
