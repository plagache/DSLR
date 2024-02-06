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
    learning_rate = 1
    # learning_rate = 1e-3

    losses = []

    delta_loss = 1
    step = 1
    steps = 10000
    # steps = 3200
    # minimal value of improvement
    precision = 4
    epsilon = 10 ** -precision
    epsilon_scientifique = 1e-4
    # epsilon_scientifique = 1e-12
    # number of step with no epsilon improvement
    patience = 100
    # while step < steps and delta_loss > epsilon:
    while step < steps:
        loss = sgd(features_tensor[step % total], neuron, class_labels[step % total], learning_rate)
        # loss = gd(features_tensor, neuron, class_labels, learning_rate)
        losses.append(loss)
        if step >= patience:
            delta_loss = losses[-patience] - losses[-1]
            delta_loss = abs(delta_loss)
        # if step % 100 == 0 or step < 100:
        if step % 100 == 0:
            print(f"step {step} loss {losses[-1]:.{precision * 2}f}")
            # print(delta_loss)
        step += 1

    weights_matrix.append(neuron.weight)
    losses_matrix.append(losses)
    print(len(losses), step)


# losses_matrix = np.array(losses_matrix)
# losses_df = pandas.DataFrame(losses_matrix.T, columns=classes)
# losses_df.to_csv("tmp/losses.csv", index=False)

weights_df = pandas.DataFrame(weights_matrix, index=classes, columns=features)
weights_df.to_csv("tmp/weights.csv", index_label="Hogwarts House")
