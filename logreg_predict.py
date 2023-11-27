import argparse
from nn import Neuron

from handle_data import classer, create_dataframe

parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
parser.add_argument('dataset', help='the dataset csv file')
parser.add_argument('weights', help='the weights csv file')
parser.add_argument('quartiles', help='the quartiles csv file')
args = parser.parse_args()

dataset = create_dataframe(args.dataset)
dataset = dataset.drop(columns="Hogwarts House")

weights = create_dataframe(args.weights)
quartiles = create_dataframe(args.quartiles)

list_quartiles = []
for row in quartiles.itertuples(index=False, name=None):
    list_quartiles.append(row)
# print(list_quartiles)

# print (weights, "\n", quartiles, "\n")


# exit()
dataset = classer(dataset, None, list_quartiles)
# print(dataset)
dataset = dataset.to_numpy()
# print(dataset)

# print(weights[0])
weights = weights.iloc[0].to_numpy()
# print(weights)
row_size = weights.size
# print(row_size)
neuron = Neuron(row_size, weights)
# print(neuron.weight)


outputs = neuron.outputs(dataset.T)
# print(outputs)
# print(len(outputs))

def decoder(outputs):
    houses=[]
    for line in outputs:
        if line > 0.5:
            houses.append("Gryffindor")
        else:
            houses.append("not")
    return houses


def save_houses(results):
    file = open("houses.csv", "w")
    line = "Index,Hogwarts House\n"
    for index, house in enumerate(results):
        line += f"{index},{house}\n"
    file.write(line)
    file.close()

results = decoder(outputs)
save_houses(results)
