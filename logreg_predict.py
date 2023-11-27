import argparse
from nn import Neuron

from handle_data import create_dataframe

parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
parser.add_argument('dataset', help='the dataset csv file')
parser.add_argument('weights', help='the weights csv file')
parser.add_argument('quartiles', help='the quartiles csv file')
args = parser.parse_args()

dataset = create_dataframe(args.dataset)
weights = create_dataframe(args.weights)
quartiles = create_dataframe(args.quartiles)


print (weights, "\n", quartiles, "\n")


dataset = dataset.to_numpy()


# print(weights[0])
weights = weights.iloc[0].to_numpy()
print(weights)
row_size = weights.size
print(row_size)
neuron = Neuron(row_size, weights)
print(neuron.weight)


outputs = neuron.outputs()


def save_houses(results):
    file = open("houses.csv", "w")
    file.write("Index,Hogwarts House\n")
    line = ""
    # for index, house in enumerate(results):
    line += "\n"
    file.write(line)
    file.close()
