import argparse

from data_preprocessing import create_dataframe, split_dataframe
from variables import sampling

def sample(dataset, rate):
    if 0 < rate < 1:
        return split_dataframe(dataset, rate)
    else:
        return dataset, dataset


if __name__ == "__main__":
    print("\n------------ Sampling -----------")
    parser = argparse.ArgumentParser(description="A simple python program to sample dataset")
    parser.add_argument("dataset", help="the dataset csv file")
    args = parser.parse_args()

    dataset = create_dataframe(args.dataset)

    test_sample, train_sample = sample(dataset, sampling)
    print(f"Sample fraction: {sampling * 100}%")

    test_sample.to_csv("datasets/dataset_test.csv")
    train_sample.to_csv("datasets/dataset_train.csv")
