import argparse

from data_preprocessing import create_classes, create_dataframe, create_labels, create_training_data, get_selected_features, split_dataframe
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


if __name__ == "__main__":
    print("\n------------ Cross Validation -----------")
    parser = argparse.ArgumentParser(description="A simple python program to perform cross validation")
    parser.add_argument("dataset", help="the dataset csv file")
    args = parser.parse_args()

    dataset = create_dataframe(args.dataset)

    classes = create_classes(dataset)

    brain = Brain(classes, selected_features)

    for fold in k_fold(dataset, number_of_fold):
        test_sample = fold
        train_sample = dataset.drop(test_sample.index)

        labels = create_labels(train_sample, classes)
        labels_tensor = labels.to_numpy().T

        train_selected = get_selected_features(train_sample, selected_features)
        x_train, _ = create_training_data(train_selected)
        features_tensor = x_train.to_numpy()

        test_selected = get_selected_features(test_sample, selected_features)
        x_test, _ = create_training_data(test_selected)

        losses, weights, accuracies = training(brain, features_tensor, labels_tensor, learning_rate, steps, False, x_test, test_sample)
        # print(losses, weights)
