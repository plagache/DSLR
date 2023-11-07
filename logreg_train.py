import numpy as np
from numpy import random
from handle_data import create_dataframe, classer
from math import isnan
import argparse

parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
parser.add_argument('filename', help='the dataset csv file')
parser.add_argument('--web', action='store_true', help='export data to html output')
args = parser.parse_args()

dataset = create_dataframe(args.filename)

ys, tensor = classer(dataset, "Gryffindor")

print(ys, tensor)

def sigmoid(z):
  if z > 0:   
    exp = np.exp(-z)
    return 1.0 / (1 + exp)
  else:
    exp = np.exp(z)
    return exp / (1 + exp)


class Neuron():

    def __init__(self, number_of_input):
        # maybe we can create some sort of a verification here
        # self.weight = np.random.rand(1, number_of_input)
        self.weight = np.zeros(number_of_input, dtype=np.float64)

    def __call__(self, inputs):
        # what we want is >> weight * inputs
        z = 0
        for weight, x in zip(self.weight, inputs):
            if not isnan(x):
                z += weight * x

        return sigmoid(z)


tensor = tensor.to_numpy()
# print("numpy array :", tensor)

row_count = len(tensor)
row_size = len(tensor[0])
neuron = Neuron(row_size)
total = 1000
learning_rate = 0.3

def count_nan_per_column(tensor):
    counts = [0] * 13
    for row in tensor:
        for index, col in enumerate(row):
            if isnan(col):
                counts[index] += 1
    return counts
nan_counts_by_column = count_nan_per_column(tensor)

for step in range(total):
    loss = 0

    #compute output of neuron and build diffs
    diffs = []
    for input, y in zip(tensor, ys):
        output = neuron(input)
        diffs.append(output - y)
        #diff = 0 (good prediction) => loss is 0
        #diff != 0 (bad prediction) => loss is -inf

        #update loss for row
        if y == 1.0:
            loss += np.log(output)
        else:
            loss += np.log(1 - output)

    loss = - loss / row_count

    # Update weight
    for index, weight in enumerate(neuron.weight):
        column_size = row_count - nan_counts_by_column[index]
        sum = 0
        # print("weight :", weight)
        for diff, element in zip(diffs, (tensor.T)[index]):
            if not isnan(element):
                sum += diff * element
        # print("mul :", mul)
        column_derivative = (1 / column_size) * sum
        # print("div :", div)
        neuron.weight[index] -= learning_rate * column_derivative

    # print(step)

    if step % 100 == 0 :
        # print(f"step : {step}| weight : {neuron.weight}| loss : {loss}| log0count {count}")
        print(f"loss : {loss}| log0count {count}")
