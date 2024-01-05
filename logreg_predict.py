import argparse
from nn import Neuron
import pandas

from handle_data import create_dataframe, robust_scale, split_by_houses

parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
parser.add_argument('dataset', help='the dataset csv file')
parser.add_argument('weights', help='the weights csv file')
parser.add_argument('quartiles', help='the quartiles csv file')
args = parser.parse_args()

dataset = create_dataframe(args.dataset)
dataset = dataset.drop(columns="Hogwarts House")

parameters = create_dataframe(args.weights)
quartiles = create_dataframe(args.quartiles)

list_quartiles = [ row for row in quartiles.itertuples(index=False, name=None) ]

# print (weights, "\n", quartiles, "\n")
# print (parameters, "\n")

dataset = robust_scale(dataset, list_quartiles)
dataset = dataset.to_numpy()


models = pandas.DataFrame()

for house in split_by_houses(parameters):
    # print(house)

    house_name = house["Hogwarts House"].iloc[0]
    # print(house_name)

    weights = house.select_dtypes(include=["float64"]).to_numpy().reshape(-1)
    # print(weights)

    neuron = Neuron(weights.size, weights)

    models[house_name] = neuron.outputs(dataset.T)

print(models)
# print(models.idxmax(axis="columns"))


prediction = models.idxmax(axis="columns")
prediction.to_csv("houses.csv", index_label="Index", header=["Hogwarts House"])
