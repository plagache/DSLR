import argparse
import pandas
from nn import Neuron
from handle_data import create_dataframe, robust_scale, split_by_houses
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
parser.add_argument('dataset', help='the dataset csv file')
parser.add_argument('weights', help='the weights csv file')
parser.add_argument('quartiles', help='the quartiles csv file')
args = parser.parse_args()

dataset = create_dataframe(args.dataset)
# dataset = dataset.drop(columns=["Care of Magical Creatures"])
# dataset = dataset.drop(columns=["Arithmancy", "Care of Magical Creatures"])
y_true = dataset["Hogwarts House"].tolist()
# test_samples = dataset.filter(["Hogwarts House"])
dataset = dataset.drop(columns="Hogwarts House")
dataset = dataset.select_dtypes(include=["float64"])

parameters = create_dataframe(args.weights)
quartiles = create_dataframe(args.quartiles)

list_quartiles = [ row for row in quartiles.itertuples(index=False, name=None) ]

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
    print(models[house_name])

print("models:\n", models)
# print(models.idxmax(axis="columns"))
# print("test samples:\n", test_samples)


prediction = models.idxmax(axis="columns")
print("prediction:\n", prediction)
prediction.to_csv("houses.csv", index_label="Index", header=["Hogwarts House"])


y_prediction = prediction.tolist()
# y_true = test_samples["Hogwarts House"].tolist()
accuracy = accuracy_score(y_true, y_prediction)
print(accuracy)
