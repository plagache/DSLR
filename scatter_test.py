import argparse
import pandas as pd

import matplotlib.pyplot as pyplot

from data_preprocessing import create_classes, create_dataframe, split_by_classes
from variables import colors

pyplot.style.use("gruvbox.mplstyle")

parser = argparse.ArgumentParser(description="A simple python program to print the scatter plots of a given csv dataset")
parser.add_argument("train_set", help="the train dataset csv file")
parser.add_argument("test_set", help="the test dataset csv file")
parser.add_argument("--show", action="store_true", help="hangs program to display plots")
args = parser.parse_args()

dataset = create_dataframe(args.train_set)

test_data = create_dataframe(args.test_set)
test_data["Hogwarts House"] = test_data["Hogwarts House"].apply(lambda x: "Test_Set")
dataset = pd.concat([dataset, test_data], ignore_index=True, sort=False)

datasets = split_by_classes(dataset)
classes = create_classes(dataset)
features = dataset.select_dtypes(include=["float64"]).columns.tolist()
dataset_by_class = list(zip(datasets, classes))

# Get a list of course pairs (Astro, Herbo)
features_pairs = [(feature, list(filter(lambda x: x != feature, features))) for feature in features]

for given_feature, other_features in features_pairs:
    for other_feature in other_features:
        title = f"{given_feature} - {other_feature}"

        pyplot.title(title)

        # set xylabels
        for dataset, class_name in dataset_by_class:
            pyplot.scatter(
                dataset[given_feature],
                dataset[other_feature],
                c=colors[class_name],
                alpha=0.6,
                label=class_name,
            )
        pyplot.legend(loc="best")

        filename = f"static/Image/scatter/{title}.png"
        pyplot.savefig(filename, format="png")
        print(f"created {filename}")

        if args.show is True and title == "Arithmancy - Care of Magical Creatures":
            pyplot.show()
        pyplot.close()
