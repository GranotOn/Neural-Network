import time


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    def add(self, layer):  # Add a layer
        self.layers.append(layer)

    def use(self, loss, loss_prime):  # set loss
        self.loss = loss
        self.loss_prime = loss_prime

    def predict(self, input_data):
        layer_input = input_data
        for layer in self.layers[:-1]:
            layer_input = layer.forward_propagation(layer_input)

        return layer_input

    def fit(self, x_train, y_train, epochs, learning_rate):  # Training method
        samples = len(x_train)
        for i in range(epochs):
            start = time.time()
            print("\n====== Epoch", i, "======")
            err = 0

            for j in range(samples):
                # fastforward
                output = x_train[j]

                loss_layer_input = self.predict(output)

                loss_layer = self.layers[-1]

                train_sample_error = loss_layer.get_loss(
                    loss_layer_input, y_train[j])

                err += train_sample_error

                layer_gradient = loss_layer.get_gradient()

                for layer in self.layers[-2::-1]:  #Reversed
                    layer_gradient = layer.backward_propagation(layer_gradient)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d error=%f' % (i + 1, epochs, err))
            print("\n Time: ", time.time() - start)
