import numpy as np
import pandas
from nn import Neuron
from optim import sgd, gd
from data_preprocessing import create_dataframe, create_classes, create_training_data, create_labels
import argparse
from tqdm import tqdm
from variables import labels_column

parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
parser.add_argument('filename', help='the dataset csv file')
parser.add_argument('--web', action='store_true', help='export data to html output')
args = parser.parse_args()

dataset = create_dataframe(args.filename)

print("\n------------ Training -----------")

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
    learning_rate = 1

    losses = []

    steps = 10000
    for step in (t:=tqdm(range(steps))):
        # loss = sgd(features_tensor[step % total], neuron, class_labels[step % total], learning_rate)
        loss = gd(features_tensor, neuron, class_labels, learning_rate)
        losses.append(loss)
        # t.set_description(f"loss:{losses[-1]:.6f}")

    weights_matrix.append(neuron.weight)
    losses_matrix.append(losses)


losses_matrix = np.array(losses_matrix)
losses_df = pandas.DataFrame(losses_matrix.T, columns=classes)
losses_df.to_csv("tmp/losses.csv", index=False)

weights_df = pandas.DataFrame(weights_matrix, index=classes, columns=features)
weights_df.to_csv("tmp/weights.csv", index_label=labels_column)
