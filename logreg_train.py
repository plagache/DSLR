import argparse

import numpy as np
import pandas
from tqdm import tqdm

from accuracy_test import test_accuracy
from data_preprocessing import create_classes, create_dataframe, create_labels, create_training_data
from logreg_predict import predict
from nn import Brain
from optim import gradient_descent, learning_rate_scheduler, stochastic_gradient_descent
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

    accuracy = False
    x_test = None
    if test_sample is not None:
        accuracy = True
        x_test = create_training_data(test_sample)

    losses = []
    accuracies = []

    learning_rate = learning_rate
    steps = steps
    for step in (t := tqdm(range(steps))):
        learning_rate = learning_rate_scheduler(learning_rate, step)
        if stochastic is True:
            loss = stochastic_gradient_descent(brain, learning_rate, features_tensor, labels_tensor)
        else:
            loss = gradient_descent(brain, learning_rate, features_tensor, labels_tensor)

        losses.append(loss)

        description = f"loss: {loss}"

        if accuracy is True:
            predictions = predict(brain, x_test)
            calculated_accuracy = test_accuracy(test_sample, predictions, labels_column)
            accuracies.append(calculated_accuracy)

            description = f"accuracy: {calculated_accuracy * 100:.2f}%"

        t.set_description(description)

    losses = np.stack(losses)
    losses_df = pandas.DataFrame(losses, columns=classes)
    losses_df.to_csv("tmp/losses.csv", index=False)

    weights_df = pandas.DataFrame(brain.weights, index=classes, columns=features)
    weights_df.to_csv("tmp/weights.csv", index_label=labels_column)

    if accuracy is True:
        accuracy_df = pandas.DataFrame(accuracies)
        accuracy_df.to_csv("tmp/accuracies.csv", index=False)
        # return losses_df, weights_df, accuracy_df

    return losses_df, weights_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
    parser.add_argument("train_set", help="the dataset csv file")
    parser.add_argument("--accuracy", action="store", help="use accuracy with given test set", dest="test_set", default=None)
    args = parser.parse_args()

    train_sample = create_dataframe(args.train_set)

    if args.test_set is None:
        training(train_sample, learning_rate, steps)
    else:
        test_sample = create_dataframe(args.test_set)
        training(train_sample, learning_rate, steps, test_sample)
