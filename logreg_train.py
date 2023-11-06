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

class Neuron():

    def __init__(self, number_of_input):
        # maybe we can create some sort of a verification here
        self.weight = [random.uniform(-1 ,1) for _ in range(number_of_input)]

    def __call__(self, inputs):
        # what we want is >> weight * inputs
        z = 0
        for weight, x in zip(self.weight, inputs):
            if not isnan(x):
                z += weight * x

        activation_function = 1 / (1 + np.exp(-z))
        return activation_function


tensor = tensor.to_numpy()
# print("numpy array :", tensor)

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
number_of_missing_per_column = count_nan_per_column(tensor)

for step in range(total):

    outputs = [neuron(input) for input in tensor]
    # print(outputs)

    diffs = []
    for x, y in zip(outputs, ys):
        diffs.append(x - y)
    # print(diffs)

    # Update weight
    size = 1600
    for index, weight in enumerate(neuron.weight):
        column_size = size - number_of_missing_per_column[index]
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
        print("step :", step, "| weight :", neuron.weight)


