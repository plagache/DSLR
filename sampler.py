import argparse

from data_preprocessing import create_dataframe, split_dataframe
from variables import sampling


if __name__ == "__main__":
    print("\n------------ Sampling -----------")
    parser = argparse.ArgumentParser(description="A simple python program to sample dataset")
    parser.add_argument("dataset", help="the dataset csv file")
    args = parser.parse_args()

    dataset = create_dataframe(args.dataset)

    if 0 < sampling < 1:
        test_sample, train_sample = split_dataframe(dataset, sampling)
    else:
        test_sample, train_sample = dataset, dataset
    print(f"Sample fraction: {sampling * 100}%")

    test_sample.to_csv("datasets/dataset_test.csv")
    train_sample.to_csv("datasets/dataset_train.csv")
