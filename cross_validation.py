import argparse

from data_preprocessing import create_classes, create_dataframe, create_labels, get_quartiles, get_selected_features, robust_scale, split_dataframe
from logreg_train import training
from nn import Brain
from variables import learning_rate, number_of_fold, selected_features, steps


def k_fold(dataframe, k):
    folds = []
    while k > 1:
        percentage = 1 / k
        fold, dataframe = split_dataframe(dataframe, percentage)
        folds.append(fold)
        k -= 1
    return folds

def cross_validation(dataset):

    classes = create_classes(dataset)

    brain = Brain(classes, selected_features)

    cross_validation_data = []

    for fold in k_fold(dataset, number_of_fold):
        test_sample = fold
        train_sample = dataset.drop(test_sample.index)

        labels = create_labels(train_sample, classes)
        labels_tensor = labels.to_numpy().T

        train_selected = get_selected_features(train_sample, selected_features)
        quartiles = get_quartiles(train_selected)
        x_train = robust_scale(train_selected, quartiles)
        features_tensor = x_train.to_numpy()

        test_selected = get_selected_features(test_sample, selected_features)
        x_test = robust_scale(test_selected, quartiles)

        losses, weights, accuracies = training(brain, features_tensor, labels_tensor, learning_rate, steps, False, x_test, test_sample)
        cross_validation_data.append({
            "losses": losses,
            "accuracies": accuracies
        })

    return cross_validation_data

if __name__ == "__main__":
    print("\n------------ Cross Validation -----------")
    parser = argparse.ArgumentParser(description="A simple python program to perform cross validation")
    parser.add_argument("dataset", help="the dataset csv file")
    args = parser.parse_args()

    dataset = create_dataframe(args.dataset)
    data = cross_validation(dataset)
    print(data)
