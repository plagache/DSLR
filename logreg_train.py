import numpy as np
from nn import Neuron
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



tensor = tensor.to_numpy()
ys = ys.to_numpy()
# print("numpy array :", tensor)
# print("numpy ys :", ys)

row_count = len(tensor)
row_size = len(tensor[0])
neuron = Neuron(row_size)
total = 500
learning_rate = 0.6

losses = []

def optimizer():
    # input: tensor, ys, neuron.weight
    # output: outputs, diffs, losses
    # z = neuron.weight @ tensor.T
    # outputs = sigmoid(z)
    # outputs = neuron(tensor.T)
    outputs = neuron.outputs(tensor.T)
    diffs = outputs - ys
    loss = 0
    for element, y in zip(outputs, ys):
        loss += y * np.log(element) + (1 - y) * np.log(1 - element)
    loss = - loss / row_count
    losses.append(loss)

    # Update weight
    # inputs: neuron.weight, diffs, tensor
    # outputs: neuron.weight.updater
    # diffs @ tensor = [13, 1]
    derivative = diffs @ tensor
    derivative /= row_count
    neuron.weight -= learning_rate * derivative


def save_weight(neuron):
    file = open("neuron_weight.csv", "w")
    file.write("w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13\n")
    line = ""
    for index, weight in enumerate(neuron.weight):
        if index > 0:
            line += ","
        line += f"{weight}"
    line += "\n"
    file.write(line)
    file.close()


for step in range(total):
    optimizer()
    if step % 100 == 0 :
        print(f"step : {step}\nloss : {losses[-1]}\nweight : {neuron.weight}\n")
save_weight(neuron)
