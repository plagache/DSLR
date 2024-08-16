import argparse

import matplotlib.pyplot as pyplot

from data_preprocessing import create_classes, create_dataframe, split_by_classes
from variables import colors, histogram_feature

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple python program to print the histogram plots of a given csv dataset")
    parser.add_argument("filename", help="the dataset csv file")
    parser.add_argument("--web", action="store_true", help="does not hang program to display plots")
    args = parser.parse_args()

    pyplot.style.use("gruvbox.mplstyle")
    dataset = create_dataframe(args.filename)


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

        if args.web is False and feature == histogram_feature:
            pyplot.show()
        pyplot.close()
