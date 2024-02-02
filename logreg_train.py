import numpy as np
import pandas
from nn import Neuron
from optim import sgd, gd
from data_preprocessing import create_dataframe, create_classes, create_training_data, create_labels
import argparse

parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
parser.add_argument('filename', help='the dataset csv file')
parser.add_argument('--web', action='store_true', help='export data to html output')
args = parser.parse_args()

dataset = create_dataframe(args.filename)

print("\n\n------------ Training -----------\n")

weights_matrix = []
losses_matrix = []

x_train = create_training_data(dataset)
features = x_train.columns.tolist()
features_tensor = x_train.to_numpy()

classes = create_classes(dataset)
labels = create_labels(dataset, classes)
labels_tensor = labels.to_numpy().T


for class_labels, class_name in zip(labels_tensor, classes):
    print(f"\n{class_name}:")

    row_size = len(features_tensor[0])
    neuron = Neuron(row_size)
    max = len(dataset)
    total = max - 1
    learning_rate = 9

    losses = []

    step = 1
    step_max = 1500
    while len(losses) == 0 or losses[-1] > 0.01 and step <= step_max:
        # loss = sgd(tensor[step % total], neuron, ys[step % total], learning_rate)
        loss = gd(features_tensor, neuron, class_labels, learning_rate)
        losses.append(loss)
        if step % 100 == 0:
            print(f"step {step} loss {losses[-1]:.4f}")
        step += 1

    weights_matrix.append(neuron.weight)
    losses_matrix.append(losses)


losses_matrix = np.array(losses_matrix)
losses_df = pandas.DataFrame(losses_matrix.T, columns=classes)
losses_df.to_csv("tmp/losses.csv", index=False)

weights_df = pandas.DataFrame(weights_matrix, index=classes, columns=features)
weights_df.to_csv("tmp/weights.csv", index_label="Hogwarts House")
