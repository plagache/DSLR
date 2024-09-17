import argparse

import pandas as pd

from data_preprocessing import create_dataframe

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple python program to print the pait plot of a given csv dataset")
    parser.add_argument("train_set", help="the train dataset csv file")
    parser.add_argument("test_set", help="the test dataset csv file")
    args = parser.parse_args()


    dataset = create_dataframe(args.train_set)

    test_data = create_dataframe(args.test_set)
    test_data["Hogwarts House"] = test_data["Hogwarts House"].apply(lambda x: "unknown")
    new_dataset = pd.concat([dataset, test_data], ignore_index=True, sort=False)
    new_dataset.to_csv("datasets/dataset_train.csv")
