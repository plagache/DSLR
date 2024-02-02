import argparse
import pandas
from nn import Neuron
from data_preprocessing import create_dataframe, robust_scale, split_by_houses

parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
parser.add_argument('dataset', help='the dataset csv file')
parser.add_argument('weights', help='the weights csv file')
parser.add_argument('quartiles', help='the quartiles csv file')
args = parser.parse_args()

dataset = create_dataframe(args.dataset)
dataset = dataset.select_dtypes(include=["float64"])

parameters = create_dataframe(args.weights)
quartiles = create_dataframe(args.quartiles)

list_quartiles = list(quartiles.itertuples(index=False, name=None))

print("\n\n------------ Predict -----------\n")
scaleddataset = robust_scale(dataset, list_quartiles).to_numpy()

models = pandas.DataFrame()

for house in split_by_houses(parameters):
    house_name = house["Hogwarts House"].iloc[0]
    weights = house.select_dtypes(include=["float64"]).to_numpy().reshape(-1)
    neuron = Neuron(weights.size, weights)
    models[house_name] = neuron.outputs(scaleddataset.T)


prediction = models.idxmax(axis="columns")
prediction.to_csv("houses.csv", index_label="Index", header=["Hogwarts House"])
