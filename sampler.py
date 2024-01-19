from data_preprocessing import create_dataframe
import argparse

parser = argparse.ArgumentParser(description="A simple python program to sample dataset")
parser.add_argument('dataset', help='the dataset csv file')
parser.add_argument('-s', '--size', default=0.1 , type=float, required=False, help='size of the sample to create')
args = parser.parse_args()

dataset = create_dataframe(args.dataset)


def split_dataframe(dataframe, test_percent: float):
    test_sample = dataframe.groupby("Hogwarts House").sample(frac=test_percent)
    train_sample = dataframe.drop(test_sample.index)
    return test_sample, train_sample


test_sample, train_sample = split_dataframe(dataset, args.size)
print(test_sample, train_sample)

test_sample.to_csv("datasets/dataset_test.csv")
train_sample.to_csv("datasets/dataset_train.csv")
