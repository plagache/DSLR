import argparse

import numpy as np
import pandas
from tqdm import tqdm

from accuracy_test import test_accuracy
from data_preprocessing import create_classes, create_dataframe, create_labels, create_training_data
from logreg_predict import predict
from nn import Brain
from optim import gd, learning_rate_scheduler, sgd
from variables import labels_column, learning_rate, steps, stochastic


def training(train_sample, learning_rate, steps, test_sample=None):
    print("\n------------ Training -----------")

    x_train = create_training_data(train_sample)
    features = x_train.columns.tolist()
    features_tensor = x_train.to_numpy()

    classes = create_classes(train_sample)
    labels = create_labels(train_sample, classes)
    labels_tensor = labels.to_numpy().T

    brain = Brain(classes, features)

    x_test = None
    samples = None
    if test_sample is not None:
        x_test = create_dataframe(test_sample)
        samples = create_training_data(x_test)

    losses = []
    accuracies = []

    learning_rate = learning_rate
    steps = steps
    for step in (t := tqdm(range(steps))):
        learning_rate = learning_rate_scheduler(learning_rate, step)
        if stochastic is True:
            loss = sgd(brain, learning_rate, features_tensor, labels_tensor)
        else:
            loss = gd(brain, learning_rate, features_tensor, labels_tensor)

        losses.append(loss)

        description = f"loss: {loss}"

        # if accuracy is True:
        if test_sample is not None:
            prediction_test = predict(brain, samples)
            calculated_accuracy = test_accuracy(x_test, prediction_test, labels_column)
            accuracies.append(calculated_accuracy)

            description = f"accuracy: {calculated_accuracy * 100:.2f}%"

        t.set_description(description)

    losses = np.stack(losses)
    losses_df = pandas.DataFrame(losses, columns=classes)
    losses_df.to_csv("tmp/losses.csv", index=False)

    weights_df = pandas.DataFrame(brain.weights, index=classes, columns=features)
    weights_df.to_csv("tmp/weights.csv", index_label=labels_column)

    # if accuracy is True:
    if test_sample is not None:
        accuracy_df = pandas.DataFrame(accuracies)
        accuracy_df.to_csv("tmp/accuracies.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
    parser.add_argument("train_set", help="the dataset csv file")
    parser.add_argument("--accuracy", action="store", help="use accuracy", dest="test_set", default=None)
    args = parser.parse_args()

    train_sample = create_dataframe(args.train_set)

    if args.test_set is None:
        training(train_sample, learning_rate, steps)
    else:
        training(train_sample, learning_rate, steps, args.test_set)
