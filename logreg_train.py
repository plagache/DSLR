import numpy as np
import pandas
from nn import Brain
from optim import sgd, gd
from data_preprocessing import create_dataframe, create_classes, create_training_data, create_labels
import argparse
from tqdm import tqdm
from variables import labels_column
from accuracy_test import test_accuracy
from sampler import split_dataframe

parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
parser.add_argument('filename', help='the dataset csv file')
parser.add_argument('--web', action='store_true', help='export data to html output')
args = parser.parse_args()

dataset = create_dataframe(args.filename)

print("\n------------ Training -----------")

weights_matrix = []
losses_matrix = []

test_sample, train_sample = split_dataframe(dataset, 0.3)
dataset = train_sample

x_train = create_training_data(dataset)
features = x_train.columns.tolist()
features_tensor = x_train.to_numpy()

classes = create_classes(dataset)
labels = create_labels(dataset, classes)
labels_tensor = labels.to_numpy().T


# for class_labels, class_name in zip(labels_tensor, classes):
#     print(f"\n{class_name}:")
#
#     row_size = len(features_tensor[0])
#     neuron = Neuron(row_size)
#     max = len(dataset)
#     total = max - 1
#     learning_rate = 2e-1
#
#     losses = []
#
#     steps = 1100
#     for step in (t:=tqdm(range(steps))):
#         # loss = sgd(features_tensor[step % total], neuron, class_labels[step % total], learning_rate)
#         # loss = sgd(features_tensor, neuron, class_labels, learning_rate)
#         loss = gd(features_tensor, neuron, class_labels, learning_rate)
#         loss = gd(features_tensor, neuron, class_labels, learning_rate)
#
#         # samples = test_sample
#         # new = create_training_data(samples)
#         # print(new)
#         # prediction = neuron.outputs(new.to_numpy().T)
#         # print(prediction)
#         # accuracy = test_accuracy(test_sample, prediction, labels_column)
#
#         losses.append(loss)
#         # t.set_description(f"loss:{losses[-1]:.6f}")
#
#     weights_matrix.append(neuron.weight)
#     losses_matrix.append(losses)


# print(labels_tensor.T, features_tensor)
brain = Brain(features_tensor, classes, features, labels_tensor)

losses = []
learning_rate = 2e-1
steps = 1100
for step in (t:=tqdm(range(steps))):
    # loss = sgd(features_tensor[step % total], neuron, class_labels[step % total], learning_rate)
    # loss = sgd(features_tensor, neuron, class_labels, learning_rate)
    loss = gd(brain, learning_rate)

    # samples = test_sample
    # new = create_training_data(samples)
    # print(new)
    # print(samples[labels_column])
    # brain_test = Brain(new, classes, features, weights=brain.weights)
    # prediction = brain_test.predictions()
    # print(prediction.T)
    # Use argmax to find the index of the maximum value in each row
    # max_indices = np.argmax(prediction.T, axis=1)
    # predicted_classes = [classes[idx] for idx in max_indices]
    # df = pandas.DataFrame({"Hogwarts House": predicted_classes})
    # print(df)
    # prediction = neuron.outputs(new.to_numpy().T)
    # print(prediction)
    # accuracy = test_accuracy(test_sample, df, labels_column)

    # t.set_description(f"accuracy: {accuracy * 100:.2f}%")
    # losses.append(loss)
    # t.set_description(f"loss:{losses[-1]:.6f}")
    # exit()

# print(brain.weights)

# losses_matrix = np.array(losses_matrix)
# losses_df = pandas.DataFrame(losses_matrix.T, columns=classes)
# losses_df.to_csv("tmp/losses.csv", index=False)

# weights_df = pandas.DataFrame(weights_matrix, index=classes, columns=features)
weights_df = pandas.DataFrame(brain.weights, index=classes, columns=features)
weights_df.to_csv("tmp/weights.csv", index_label=labels_column)
