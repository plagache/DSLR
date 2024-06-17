import argparse
import pandas as pd

import matplotlib.pyplot as pyplot

from data_preprocessing import create_classes, create_dataframe, split_by_classes
from variables import colors

parser = argparse.ArgumentParser(description="A simple python program to print the histogram plots of a given csv dataset")
parser.add_argument("train_set", help="the train dataset csv file")
parser.add_argument("test_set", help="the test dataset csv file")
parser.add_argument("--show", action="store_true", help="hangs program to display plots")
args = parser.parse_args()

pyplot.style.use("gruvbox.mplstyle")
dataset = create_dataframe(args.train_set)

test_data = create_dataframe(args.test_set)
test_data["Hogwarts House"] = test_data["Hogwarts House"].apply(lambda x: "Test_Set")
dataset = pd.concat([dataset, test_data], ignore_index=True, sort=False)

datasets = split_by_classes(dataset)
classes = create_classes(dataset)
features = dataset.select_dtypes(include=["float64"]).columns.tolist()
dataset_by_class = list(zip(datasets, classes))

for feature in features:
    pyplot.title(feature)

    for dataset, class_name in dataset_by_class:
        pyplot.hist(dataset[feature], color=colors[class_name], alpha=0.5, label=class_name)

    pyplot.legend(loc="best")

    filename = f"static/Image/hist/{feature}.png"
    pyplot.savefig(filename, format="png")
    print(f"created {filename}")

    if args.show is True and feature == "Care of Magical Creatures":
        pyplot.show()
    pyplot.close()
