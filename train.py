import csv
import numpy as np
import pickle

from Network import Network
from FCLayer import FCLayer
from ActivationLayer import ActivationLayer
from utils import tanh, tanh_prime, mse, mse_prime

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
x_train = x_train.reshape(x_train.shape[0], 1, 3072)
x_train = x_train.astype('float32')

y_train = np.array(output)

# network
net = Network()
net.add(FCLayer(3072, 1541))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(1541, 775))
net.add(ActivationLayer(tanh, tanh_prime))
net.add(FCLayer(775, 10))
net.add(ActivationLayer(tanh, tanh_prime))

# train
net.use(mse, mse_prime)

net.fit(x_train, y_train, epochs=40, learning_rate=0.02)

# Save model

model_file = open("network_model", 'wb')
pickle.dump(net, model_file)
model_file.close()