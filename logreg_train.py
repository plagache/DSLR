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
        # maybe we can create some sort of a verification here
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
for step in range(total):
    count = 0
    loss = 0

    #compute output of neuron and build diffs
    diffs = []
    for input, y in zip(tensor, ys):
        output = neuron(input)
        diffs.append(output - y)

        #update loss for row
        if y == 1.0:
            loss += np.log(output)
        else:
            loss += np.log(1 - output)

    loss = - loss / row_count
    losses.append(loss)

    # Update weight
    for index, weight in enumerate(neuron.weight):
        sum = 0
        # print("weight :", weight)
        for diff, element in zip(diffs, (tensor.T)[index]):
            sum += diff * element
        # print("mul :", mul)
        column_derivative = (1 / row_count) * sum
        # print("div :", div)
        neuron.weight[index] -= learning_rate * column_derivative

    if step % 100 == 0 :
        # print(f"step : {step}| weight : {neuron.weight}| loss : {loss}| log0count {count}")
        print(f"loss : {loss}| log0count {count}")
