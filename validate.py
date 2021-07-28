import csv
import pickle
import numpy as np

validate_file = "validate.csv"


output = []
feature_set = []

with open(validate_file) as csv_file:
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

x_validate = np.array(feature_set)
#x_validate = x_validate.reshape(x_validate.shape[0], 1, 3072)
x_validate = x_validate.reshape(x_validate.shape[0], 3, 32, 32)
x_validate = x_validate.astype('float32')

y_validate = np.array(output)

# get network
model_file = open("network_model", "rb")
net = pickle.load(model_file)
model_file.close()

# validate
out = net.predict(x_validate)

correct = 0
overall = len(out)

for idx, row in enumerate(out):
    print(np.argmax(row), np.argmax(y_validate[idx]))
    if (y_validate[idx][np.argmax(row)] == 1):
        correct += 1

print('%f%%' % ((correct / overall) * 100))