import csv
import pickle
import numpy as np

test_file = "test.csv"
model = "model.p"

output = []
feature_set = []

with open(test_file) as csv_file:
    # Open CSV
    csv_reader = csv.reader(csv_file, delimiter=",")
    # Traverse rows
    for row in csv_reader:
        set = []
        for index, value in enumerate(row):
            # First column is for results
            if index == 0:
                continue
            else:
                set.append(float(value))

        feature_set.append(set)

x_test = np.array(feature_set)
x_test = x_test.reshape(x_test.shape[0], 1, 3072)
x_test = x_test.astype('float32')

# get network
model_file = open("network_model", "rb")
net = pickle.load(model_file)
model_file.close()

out = net.predict(x_test)

output = open("output.txt", "w")

for row in out:
    output.write(str((np.argmax(row) + 1)) + "\n")
output.close()
