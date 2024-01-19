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
# dataset = dataset.groupby("Hogwarts House").apply(lambda x: x)
# dataset = dataset.drop(columns=["Care of Magical Creatures"])
# dataset = dataset.drop(columns=["Arithmancy", "Care of Magical Creatures"])
# test_samples = dataset.filter(["Hogwarts House"])
dataset = dataset.drop(columns="Hogwarts House")
dataset = dataset.select_dtypes(include=["float64"])

parameters = create_dataframe(args.weights)
quartiles = create_dataframe(args.quartiles)

list_quartiles = [row for row in quartiles.itertuples(index=False, name=None)]

# print ("\n", quartiles, "\n")
# print ("\n", list_quartiles, "\n")
# print (parameters, "\n")

scaleddataset = robust_scale(dataset, list_quartiles).to_numpy()

models = pandas.DataFrame()

for house in split_by_houses(parameters):
    # print(house)

    house_name = house["Hogwarts House"].iloc[0]
    # print(house_name)

    weights = house.select_dtypes(include=["float64"]).to_numpy().reshape(-1)
    # print(weights)

    neuron = Neuron(weights.size, weights)
    # print(neuron.weight)
    # print(len(neuron.weight))

    # print(scaleddataset)
    models[house_name] = neuron.outputs(scaleddataset.T)
    # print((np.min(neuron._z), np.max(neuron._z)), sep='\n')
    # print((np.min(neuron._outputs), np.max(neuron._outputs)), sep='\n')
    # print('\n')

print("models:\n", models)
# print(models.idxmax(axis="columns"))
# print("test samples:\n", test_samples)


prediction = models.idxmax(axis="columns")
print("prediction:\n", prediction)
prediction.to_csv("houses.csv", index_label="Index", header=["Hogwarts House"])
