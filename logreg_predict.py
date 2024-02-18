import argparse

import numpy as np
import pandas

from data_preprocessing import create_dataframe, remove_unselected_features, robust_scale
from nn import Brain
from variables import labels_column, prediction_file, unselected_features


def predict(brain: Brain, inputs):
    prediction = brain.predictions(inputs)
    max_indices = np.argmax(prediction.T, axis=1)
    prediction = [brain.classes[index] for index in max_indices]
    prediction = pandas.DataFrame({"Hogwarts House": prediction})
    return prediction


if __name__ == "__main__":
    print("\n------------ Predict -----------")
    parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
    parser.add_argument("dataset", help="the dataset csv file")
    parser.add_argument("weights", help="the weights csv file")
    parser.add_argument("quartiles", help="the quartiles csv file")
    args = parser.parse_args()

    dataset = create_dataframe(args.dataset)
    dataset = dataset.select_dtypes(include=["float64"])
    dataset = remove_unselected_features(dataset, unselected_features)

    parameters = create_dataframe(args.weights)
    quartiles = create_dataframe(args.quartiles)

    list_quartiles = list(quartiles.itertuples(index=False, name=None))

    scaleddataset = robust_scale(dataset, list_quartiles).to_numpy()

    classes = parameters[labels_column].tolist()
    parameters = parameters.set_index(labels_column)
    features = parameters.columns.tolist()

    brain = Brain(classes, features, weights=parameters.to_numpy())

    prediction = predict(brain, scaleddataset)
    prediction.to_csv(prediction_file, index_label="Index", header=[labels_column])
