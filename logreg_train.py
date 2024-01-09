import numpy as np
import pandas
from nn import Neuron
from optim import sgd, gd
from handle_data import create_dataframe, classer, split_dataframe
import argparse

parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
parser.add_argument('filename', help='the dataset csv file')
parser.add_argument('--web', action='store_true', help='export data to html output')
args = parser.parse_args()

dataset = create_dataframe(args.filename)
test_sample, dataset = split_dataframe(dataset, 0.10)
print("test sample", len(test_sample.index), "train sample", len(dataset.index))


houses = ["Gryffindor", "Ravenclaw", "Slytherin", "Hufflepuff"]
columns_name = [f"w{index}" for index in range(1, 14)]
weights_matrix = []
losses_matrix = []
for index, house in enumerate(houses):
    print(f"\n{house}:")
    # Merge ys at the end of tensor
    ys, tensor = classer(dataset, house)
    # print(ys, tensor)

    tensor = tensor.to_numpy()
    ys = ys.to_numpy()
    # print("numpy array :", tensor)
    # print("numpy ys :", ys)

    row_size = len(tensor[0])
    neuron = Neuron(row_size)
    # sgd can go on each example not furter
    max = len(dataset)
    total = max - 1
    # total = 600
    # learning_rate = 0.6
    learning_rate = 0.001
    # learning_rate = 0.004

    losses = []

    for step in range(1, (total + 1)):
        # here we can shuffle a random part of our tensor to make SGD
        loss = sgd(tensor[step], neuron, ys[step], learning_rate)
        # loss = gd(tensor, neuron, ys, learning_rate)
        losses.append(loss)
        if step % 100 == 0:
            print(f"step {step} loss {losses[-1]:.4f}")
            # print(f"weight : {neuron.weight}\n")

    weights_matrix.append(neuron.weight)
    losses_matrix.append(losses)


losses_matrix = np.array(losses_matrix)
losses_df = pandas.DataFrame(losses_matrix.T, columns=houses)
losses_df.to_csv("tmp/losses.csv", index=False)

weights_df = pandas.DataFrame(weights_matrix, index=houses, columns=columns_name)
weights_df.to_csv("tmp/weights.csv", index_label="Hogwarts House")
