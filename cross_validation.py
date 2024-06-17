import argparse

from data_preprocessing import create_dataframe, split_dataframe
from variables import number_of_fold


def k_fold(dataframe, k):
    folds = []
    # cent = 100
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

    print(k_fold(dataset, number_of_fold))
    print(dataset)
    # test_sample.to_csv("datasets/dataset_test.csv")
    # train_sample.to_csv("datasets/dataset_train.csv")
