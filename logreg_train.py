import argparse

import numpy as np
import pandas
from tqdm import tqdm

from accuracy_test import test_accuracy
from data_preprocessing import (
    create_classes,
    create_dataframe,
    create_labels,
    get_quartiles,
    get_selected_features,
    robust_scale,
)
from logreg_predict import predict
from nn import Brain
from optim import gradient_descent, learning_rate_scheduler, stochastic_gradient_descent
from variables import labels_column, learning_rate, selected_features, steps, stochastic


def training(brain, features_tensor, labels_tensor, learning_rate, steps, stochastic=False, test_features=None, test_labels=None):
    print("\n------------ Training -----------")

    losses = []
    accuracies = []

    for step in (t := tqdm(range(steps))):
        learning_rate = learning_rate_scheduler(learning_rate, step)
        if stochastic is True:
            loss = stochastic_gradient_descent(brain, learning_rate, features_tensor, labels_tensor)
        else:
            loss = gradient_descent(brain, learning_rate, features_tensor, labels_tensor)

        losses.append(loss)

        description = f"loss: {loss}"

        if test_features is not None:
            predictions = predict(brain, test_features)
            calculated_accuracy = test_accuracy(test_labels, predictions, labels_column)
            accuracies.append(calculated_accuracy)

            description = f"accuracy: {calculated_accuracy * 100:.2f}%"

        t.set_description(description)

    losses = np.stack(losses)

    return losses, brain.weights, accuracies


def save_training_data(quartiles, losses, weights, accuracies=None):
    pandas.DataFrame(quartiles, columns=["Courses", "Q1", "Q2", "Q3"]).to_csv("tmp/quartiles.csv", index=False)

    pandas.DataFrame(losses, columns=classes).to_csv("tmp/losses.csv", index=False)

    pandas.DataFrame(weights, index=classes, columns=selected_features).to_csv("tmp/weights.csv", index_label=labels_column)

    if accuracies is not None:
        pandas.DataFrame(accuracies).to_csv("tmp/accuracies.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
    parser.add_argument("train_set", help="the dataset csv file")
    parser.add_argument("--accuracy", action="store", help="use accuracy with given test set", dest="test_set", default=None)
    args = parser.parse_args()

    train_sample = create_dataframe(args.train_set)
    train_selected = get_selected_features(train_sample, selected_features)
    quartiles = get_quartiles(train_selected)
    x_train = robust_scale(train_selected, quartiles)
    features_tensor = x_train.to_numpy()

    classes = create_classes(train_sample)
    labels = create_labels(train_sample, classes)
    labels_tensor = labels.to_numpy().T

    brain = Brain(classes, selected_features)

    if args.test_set is None:
        losses, weights, accuracies = training(brain, features_tensor, labels_tensor, learning_rate, steps, stochastic)
        save_training_data(quartiles, losses, weights)
    else:
        test_sample = create_dataframe(args.test_set)
        test_selected = get_selected_features(test_sample, selected_features)
        x_test = robust_scale(test_selected, quartiles)

        losses, weights, accuracies = training(brain, features_tensor, labels_tensor, learning_rate, steps, stochastic, x_test, test_sample)
        save_training_data(quartiles, losses, weights, accuracies)
