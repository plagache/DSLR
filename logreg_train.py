import numpy as np
import pandas
from nn import Neuron
from optim import sgd, gd
from data_preprocessing import create_dataframe, classer, split_dataframe
import argparse

parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
parser.add_argument('filename', help='the dataset csv file')
parser.add_argument('--web', action='store_true', help='export data to html output')
args = parser.parse_args()

dataset = create_dataframe(args.filename)
test_sample, dataset = split_dataframe(dataset, 0.1)
# dataset = dataset.drop(columns=["Care of Magical Creatures"])
# dataset = dataset.drop(columns=["Arithmancy", "Care of Magical Creatures"])
# print("test sample", len(test_sample.index), "train sample", len(dataset.index))
# print("test sample\n", test_sample, "train sample\n", dataset)


houses = ["Gryffindor", "Ravenclaw", "Slytherin", "Hufflepuff"]
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
    # total = 3000
    # total = 600
    # learning_rate = 3
    learning_rate = 9
    # learning_rate = 0.6
    # learning_rate = 0.001
    # learning_rate = 0.004

    losses = []

    # for step in range(1, (total + 1)):
    step = 1
    while len(losses) == 0 or losses[-1] > 0.01 and step <= 1500:
        # here we can shuffle a random part of our tensor to make SGD
        # loss = sgd(tensor[step % total], neuron, ys[step % total], learning_rate)
        # print(step % total)
        # print(loss)
        loss = gd(tensor, neuron, ys, learning_rate)
        losses.append(loss)
        if step % 100 == 0:
            print(f"step {step} loss {losses[-1]:.4f}")
            # print((np.min(neuron._z), np.max(neuron._z)), sep='\n')
            # print((np.min(neuron._outputs), np.max(neuron._outputs)), sep='\n')
            # print('\n')
            # print(f"weight : {neuron.weight}\n")
        step += 1

    weights_matrix.append(neuron.weight)
    losses_matrix.append(losses)


losses_matrix = np.array(losses_matrix)
losses_df = pandas.DataFrame(losses_matrix.T, columns=houses)
losses_df.to_csv("tmp/losses.csv", index=False)

columns_name = [f"w{index}" for index in range(1, len(weights_matrix[0]) + 1)]
weights_df = pandas.DataFrame(weights_matrix, index=houses, columns=columns_name)
weights_df.to_csv("tmp/weights.csv", index_label="Hogwarts House")

test_sample.to_csv("tmp/test_sample.csv")
