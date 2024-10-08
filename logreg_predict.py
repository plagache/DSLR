import argparse

import numpy as np
import pandas

from data_preprocessing import create_dataframe, get_selected_features, robust_scale
from nn import Brain
from variables import labels_column, selected_features


def predict(brain: Brain, inputs):
    prediction = brain.predictions(inputs)
    max_indices = np.argmax(prediction.T, axis=1)
    prediction = [brain.classes[index] for index in max_indices]
    prediction = pandas.DataFrame({labels_column: prediction})
    return prediction


if __name__ == "__main__":
    print("\n------------ Predict -----------")
    parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
    parser.add_argument("dataset", help="the dataset csv file")
    parser.add_argument("weights", help="the weights csv file")
    parser.add_argument("quartiles", help="the quartiles csv file")
    args = parser.parse_args()

    dataset = create_dataframe(args.dataset)
    selected_dataset = get_selected_features(dataset, selected_features)

    parameters = create_dataframe(args.weights)
    quartiles = create_dataframe(args.quartiles)

    list_quartiles = list(quartiles.itertuples(index=False, name=None))

    scaleddataset = robust_scale(selected_dataset, list_quartiles).to_numpy()

    classes = parameters[labels_column].tolist()
    parameters = parameters.set_index(labels_column)

    brain = Brain(classes, selected_features, weights=parameters.to_numpy())

    prediction = predict(brain, scaleddataset)
    prediction.to_csv("houses.csv", index_label="Index", header=[labels_column])
