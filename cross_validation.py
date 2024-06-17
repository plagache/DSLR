import argparse

from data_preprocessing import create_dataframe, split_dataframe
from logreg_train import training
from variables import number_of_fold
from variables import learning_rate, steps


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

    folds = k_fold(dataset, number_of_fold)
    for fold in folds:
        test_sample = fold
        train_sample = dataset.drop(test_sample.index)
        # print(test_sample, train_sample)
        losses, weights = training(train_sample, learning_rate, steps, test_sample)
        print(weights)
        # print(losses, weights)
