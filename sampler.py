import argparse

from data_preprocessing import create_dataframe
from variables import labels_column, sampling


def split_dataframe(dataframe, test_percent: float):
    test_sample = dataframe.groupby(labels_column).sample(frac=test_percent)
    train_sample = dataframe.drop(test_sample.index)
    return test_sample, train_sample


if __name__ == "__main__":
    print("\n------------ Sampling -----------")
    parser = argparse.ArgumentParser( description="A simple python program to sample dataset")
    parser.add_argument("dataset", help="the dataset csv file")
    args = parser.parse_args()

    dataset = create_dataframe(args.dataset)

    if 0 < sampling < 1:
        test_sample, train_sample = split_dataframe(dataset, sampling)
    else:
        test_sample, train_sample = dataset, dataset
    print("Sample fraction:", sampling)

    test_sample.to_csv("datasets/dataset_test.csv")
    train_sample.to_csv("datasets/dataset_train.csv")
