import numpy as np
from numpy import random
from handle_data import create_dataframe, classer
import argparse

parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
parser.add_argument('filename', help='the dataset csv file')
parser.add_argument('--web', action='store_true', help='export data to html output')
args = parser.parse_args()

dataset = create_dataframe(args.filename)

ys, tensor = classer(dataset, "Gryffindor")

# print(ys, tensor)

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


class Neuron():

    def __init__(self, number_of_input):
        # self.weight = np.random.rand(1, number_of_input)
        self.weight = np.zeros(number_of_input, dtype=np.float64)

    def __call__(self, inputs):
        # what we want is >> weight * inputs
        z = 0
        for weight, x in zip(self.weight, inputs):
            z += weight * x

        return sigmoid(z)


tensor = tensor.to_numpy()
ys = ys.to_numpy()
# print("numpy array :", tensor)

row_count = len(tensor)
row_size = len(tensor[0])
neuron = Neuron(row_size)
total = 500
learning_rate = 0.6

losses = []

def optimizer():
    # input: tensor, ys, neuron.weight
    # output: outputs, diffs, losses
    z = neuron.weight @ tensor.T
    outputs = sigmoid(z)
    diffs = outputs - ys
    loss = 0
    for element, y in zip(outputs, ys):
        if y == 1.0:
            loss += np.log(element)
        else:
            loss += np.log(1 - element)
    loss = - loss / row_count
    losses.append(loss)

    # Update weight
    # inputs: neuron.weight, diffs, tensor
    # outputs: neuron.weight.updater
    # diffs @ tensor = [13, 1]
    derivative = diffs @ tensor
    derivative /= row_count
    neuron.weight -= learning_rate * derivative


for step in range(total):
    optimizer()
    if step % 100 == 0 :
        print(f"step : {step}| weight : {neuron.weight}| loss : {losses[-1]}")
