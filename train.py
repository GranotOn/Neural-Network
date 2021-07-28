from LossLayer import LossLayer
from SoftmaxLayer import SoftmaxLayer
from MaxPoolLayer import MaxPoolLayer
import csv
import numpy as np
import pickle

from Network import Network
from FCLayer import FCLayer
from CONVLayer import CONVLayer
from ActivationLayer import ActivationLayer
from FlattenLayer import FlattenLayer
from utils import relu, relu_prime, mse, mse_prime, tanh, tanh_prime, sigmoid, sigmoid_prime, cross_entropy, cross_entropy_prime

train_file = "train.csv"

# Train data

feature_set = []
output = []

# Read train file, append labels and features.
with open(train_file) as csv_file:
    # Open CSV
    csv_reader = csv.reader(csv_file, delimiter=",")
    # Traverse rows
    for row in csv_reader:
        set = []
        channel = []
        for index, value in enumerate(row):
            # First column is for results
            if index == 0:
                label_array = [0 for i in range(10)]
                label_array[int(value) - 1] = 1
                output.append(label_array)
            else:
                set.append(float(value))

        feature_set.append(set)

x_train = np.array(feature_set)
x_train = x_train.reshape(x_train.shape[0], 3, 32, 32)
# for testing the program, we shortened the training data
x_train = x_train[0:4000, 0:3, 0:32, 0:32]
x_train = x_train.astype('float32')
y_train = np.array(output)

model = 'a'
while model != 'c' and model != 'n':
    # Add code to ask whether to upload old network
    print("Continue training from existing model: press c")
    print("Start training a new model: press n")
    model = input()

if model == 'n':
    # starting a new network
    net = Network()
    net.add(CONVLayer((3, 32, 32), (3, 3), 64))
    net.add(ActivationLayer(relu, relu_prime))
    net.add(MaxPoolLayer())

    net.add(CONVLayer((64, 16, 16), (3, 3), 128))
    net.add(ActivationLayer(relu, relu_prime))
    net.add(MaxPoolLayer())

    net.add(CONVLayer((128, 8, 8), (3, 3), 256))
    net.add(ActivationLayer(relu, relu_prime))
    net.add(MaxPoolLayer())

    net.add(CONVLayer((256, 4, 4), (3, 3), 512))
    net.add(ActivationLayer(relu, relu_prime))
    net.add(MaxPoolLayer())

    net.add(FlattenLayer())

    net.add(FCLayer(2**11, 128))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))

    net.add(FCLayer(128, 256))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))

    net.add(FCLayer(256, 512))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))

    net.add(FCLayer(512, 1024))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))

    net.add(FCLayer(1024, 10))

    net.add(SoftmaxLayer())
    net.add(LossLayer(cross_entropy, cross_entropy_prime))

elif model == 'c':
    # get network
    model_file = open("network_model", "rb")
    net = pickle.load(model_file)
    model_file.close()

# train
net.use(mse, mse_prime)
net.fit(x_train, y_train, epochs=3, learning_rate=0.01)

# Save model

model_file = open("network_model", 'wb')
pickle.dump(net, model_file)
model_file.close()