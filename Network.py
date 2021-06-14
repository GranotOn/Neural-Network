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
        samples = len(input_data)
        result = []

        for i in range(samples):  # fastforward
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train, epochs, learning_rate):  # Training method
        samples = len(x_train)
        for i in range(epochs):
            start = time.time()
            print("\n====== Epoch", i, "======")
            err = 0

            for j in range(samples):
                # fastforward
                output = x_train[j]

                for layer in self.layers:
                    output = layer.forward_propagation(output)

                if (j < 5):
                    print('%d : %s - %s' % (j, output, y_train[j]))

                # compute loss

                err += self.loss(y_train[j], output)

                # back propagation

                error = self.loss_prime(y_train[j], output)

                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            learning_rate *= 0.7
            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d error=%f' % (i + 1, epochs, err))
            print("\n Time: ", time.time() - start)
