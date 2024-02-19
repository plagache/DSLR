import argparse

import numpy as np
import pandas
from tqdm import tqdm

from accuracy_test import test_accuracy
from data_preprocessing import create_classes, create_dataframe, create_labels, create_training_data
from logreg_predict import predict
from nn import Brain
from optim import gd, sgd
from variables import labels_column, learning_rate, steps

parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
parser.add_argument("train_set", help="the dataset csv file")
parser.add_argument("test_set", help="the dataset csv file")
args = parser.parse_args()

train_sample = create_dataframe(args.train_set)
test_sample = create_dataframe(args.test_set)

print("\n------------ Training -----------")

samples = create_training_data(test_sample)

x_train = create_training_data(train_sample)
features = x_train.columns.tolist()
features_tensor = x_train.to_numpy()

classes = create_classes(train_sample)
labels = create_labels(train_sample, classes)
labels_tensor = labels.to_numpy().T

brain = Brain(classes, features)

losses = []
learning_rate = learning_rate
steps = steps
for step in (t := tqdm(range(steps))):
    # loss = sgd(brain, learning_rate, features_tensor, labels_tensor)
    loss = gd(brain, learning_rate, features_tensor, labels_tensor)

    losses.append(loss)

    prediction_test = predict(brain, samples)
    accuracy = test_accuracy(test_sample, prediction_test, labels_column)

    t.set_description(f"accuracy: {accuracy * 100:.2f}%")

losses = np.stack(losses)
losses_df = pandas.DataFrame(losses, columns=classes)
losses_df.to_csv("tmp/losses.csv", index=False)

weights_df = pandas.DataFrame(brain.weights, index=classes, columns=features)
weights_df.to_csv("tmp/weights.csv", index_label=labels_column)
