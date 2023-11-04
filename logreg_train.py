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
        # print("inputs :", inputs)
        for weight, x in zip(self.weight, inputs):
            z += weight * x

        # print("z :", z)
        activation_function = 1 / (1 + np.exp(-z))
        # print(activation_function)
        # activation_function = 1 / (1 + exp(-z))
        return activation_function

    def parameters(self):
        return self.weight

# print(Tensor)
# print(cleaned.index(2))
# print(cleaned.iloc[0])
# one_row = cleaned.iloc[0]
# print(one_row.tolist())
# print(cleaned[4])

tensor = tensor.to_numpy()
# print("numpy array :", tensor)

neuron = Neuron(len(tensor[0]))

# we do not need parameters because we have only weight
# with a bias or other params we would create a new list of all params
# print(neuron.parameters())
# print("weight :", neuron.weight)

# print(tensor._values)
# print(tensor.__array__)
# print(tensor._series)
# print(tensor._series)
# print(neuron(tensor._values))


# for index in tensor():
# for line in tensor.__array__:
    # print(line)
    # print(tensor[index])
    # print(tensor._values)
    # neuron(tensor._values)
    # print(index[1])
    # print(list(index))
outputs = [neuron(input) for input in tensor]
# print(outputs)

diffs = []
for x, y in zip(outputs, ys):
    diffs.append(x - y)
print(diffs)

i = 0
mul = 0
m = 0
for weight in neuron.weight:
    print("weight :", weight)
    i += 1
    for diff, element in zip(diffs, tensor[i]):
        m += 1
        mul += diff * element
    print("mul :", mul)
    div = mul * (1 / m)
    print("div :", div)

# print("weight :", neuron.weight)
# print("exp :", np.exp(-4352))
    # neuron(index[1])
    # print("index :", index)

    # dp = 0
    # dj = 0
    # z = 0
    # i = 0


# print(len(tensor.columns))



# print(Tensor.columns)



# print(index)
# print(y)
# print(len(classer))
# jeme = dj / ft_count(Tensor["Arithmancy"])
# print(jeme)
