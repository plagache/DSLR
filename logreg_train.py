import argparse

import numpy as np
import pandas
from tqdm import tqdm

from accuracy_test import test_accuracy
from data_preprocessing import (
    create_classes,
    create_dataframe,
    create_labels,
    create_training_data,
    remove_unselected_features,
    select_numerical_features,
)
from logreg_predict import predict
from nn import Brain
from optim import gradient_descent, learning_rate_scheduler, stochastic_gradient_descent
from variables import labels_column, learning_rate, steps, stochastic, unselected_features


def training(brain, features_tensor, labels_tensor, learning_rate, steps, stochastic, test_features=None, test_labels=None):
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


def save_training_data(losses, weights, accuracies=None):
    losses_df = pandas.DataFrame(losses, columns=classes)
    losses_df.to_csv("tmp/losses.csv", index=False)

    weights_df = pandas.DataFrame(weights, index=classes, columns=features)
    weights_df.to_csv("tmp/weights.csv", index_label=labels_column)

    if accuracies is not None:
        accuracy_df = pandas.DataFrame(accuracies)
        accuracy_df.to_csv("tmp/accuracies.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
    parser.add_argument("train_set", help="the dataset csv file")
    parser.add_argument("--accuracy", action="store", help="use accuracy with given test set", dest="test_set", default=None)
    args = parser.parse_args()

    train_sample = create_dataframe(args.train_set)
    numerical_features = select_numerical_features(train_sample)
    selected_features = remove_unselected_features(numerical_features, unselected_features)
    x_train, _ = create_training_data(selected_features)
    features = x_train.columns.tolist()
    features_tensor = x_train.to_numpy()

    classes = create_classes(train_sample)
    labels = create_labels(train_sample, classes)
    labels_tensor = labels.to_numpy().T

    brain = Brain(classes, features)

    if args.test_set is None:
        losses, weights, accuracies = training(brain, features_tensor, labels_tensor, learning_rate, steps, stochastic)
        save_training_data(losses, weights)
    else:
        test_sample = create_dataframe(args.test_set)
        numerical_features = select_numerical_features(test_sample)
        selected_features = remove_unselected_features(numerical_features, unselected_features)
        x_test, _ = create_training_data(selected_features)
        losses, weights, accuracies = training(brain, features_tensor, labels_tensor, learning_rate, steps, stochastic, x_test, test_sample)
        save_training_data(losses, weights, accuracies)
