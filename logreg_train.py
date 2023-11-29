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


def optimizer(tensor, neuron, ys):
    # input: tensor, ys, neuron.weight
    # output: outputs, diffs, losses
    row_count = len(tensor)
    loss = 0

    outputs = neuron.outputs(tensor.T)
    diffs = outputs - ys

    for element, y in zip(outputs, ys):
        loss += y * np.log(element) + (1 - y) * np.log(1 - element)
    loss = - loss / row_count

    # Update weight
    # inputs: neuron.weight, diffs, tensor
    # outputs: neuron.weight.updater
    # diffs @ tensor = [13, 1]
    derivative = diffs @ tensor
    derivative /= row_count
    neuron.weight -= learning_rate * derivative
    return loss


def save_weight(neuron, house, header):
    file = open("weights.csv", "a")
    first_line = "Hogwarts House"
    line = house
    for index, weight in enumerate(neuron.weight):
        first_line += f",w{index}"
        line += f",{weight}"
    first_line += "\n"
    line += "\n"
    if header == True:
        file.write(first_line + line)
    else:
        file.write(line)
    file.close()


open('weights.csv', 'w').close()

houses = ["Gryffindor", "Ravenclaw", "Slytherin", "Hufflepuff"]
for index, house in enumerate(houses):
    ys, tensor = classer(dataset, house)
    print(ys, tensor)

    header = True if index == 0 else False

    tensor = tensor.to_numpy()
    ys = ys.to_numpy()
    # print("numpy array :", tensor)
    # print("numpy ys :", ys)

    row_size = len(tensor[0])
    neuron = Neuron(row_size)
    total = 500
    learning_rate = 0.6

    losses = []

    for step in range(total):
        loss = optimizer(tensor, neuron, ys)
        losses.append(loss)
        if step % 100 == 0 :
            print(f"step : {step}\nloss : {losses[-1]}\nweight : {neuron.weight}\n")
    save_weight(neuron, house, header)
