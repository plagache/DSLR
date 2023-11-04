import numpy as np
from numpy import random
from handle_data import create_dataframe, classer
# from math import exp
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
            z += weight * x

        activation_function = 1 / (1 + np.exp(-z))
        return activation_function


tensor = tensor.to_numpy()
# print("numpy array :", tensor)

neuron = Neuron(len(tensor[0]))
total = 10000
learning_rate = 0.3

for step in range(total):

    outputs = [neuron(input) for input in tensor]
    # print(outputs)

    diffs = []
    for x, y in zip(outputs, ys):
        diffs.append(x - y)
    # print(diffs)

    # Update weight
    i = 0
    mul = 0
    m = 0
    for weight in neuron.weight:
        # print("weight :", weight)
        for diff, element in zip(diffs, tensor[i]):
            m += 1
            mul += diff * element
        # print("mul :", mul)
        div = mul * (1 / m)
        # print("div :", div)
        neuron.weight[i] -= learning_rate * div
        i += 1

    # print(step)

    if step % 100 == 0 :
        print("step :", step, "| weight :", neuron.weight)
