import argparse
import pandas
from nn import Neuron
from data_preprocessing import create_dataframe, robust_scale
from variables import labels_column

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

scaleddataset = robust_scale(dataset, list_quartiles).to_numpy()

models = pandas.DataFrame()

parameters = parameters.set_index(labels_column)

for class_name, weights in parameters.iterrows():
    weights = weights.to_numpy().reshape(-1)
    neuron = Neuron(weights.size, weights)
    models[class_name] = neuron.outputs(scaleddataset.T)


prediction = models.idxmax(axis="columns")
prediction.to_csv("houses.csv", index_label="Index", header=[labels_column])
