import argparse

import matplotlib.pyplot as pyplot
import pandas as pd
import seaborn

from data_preprocessing import create_dataframe
from variables import colors, labels_column

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple python program to print the pait plot of a given csv dataset")
    parser.add_argument("train_set", help="the train dataset csv file")
    parser.add_argument("test_set", help="the test dataset csv file")
    parser.add_argument("--web", action="store_true", help="does not hang program to display plots")
    args = parser.parse_args()


    pyplot.style.use("gruvbox.mplstyle")
    seaborn.set_theme(style="dark")

    dataset = create_dataframe(args.train_set)

    test_data = create_dataframe(args.test_set)
    test_data["Hogwarts House"] = test_data["Hogwarts House"].apply(lambda x: "Test_Set")
    dataset = pd.concat([dataset, test_data], ignore_index=True, sort=False)


    features = dataset.select_dtypes(include=["float64"]).columns.tolist()

    features.sort()
    splot = seaborn.pairplot(
        dataset,
        vars=features,
        hue=labels_column,
        palette=colors,
        diag_kind="hist",
        plot_kws=dict(marker=".", alpha=0.8, sizes=5),
    )

    filename = "static/Image/pair/pairplot.png"
    splot.savefig(filename)
    print(f"created {filename}")

    if args.web is False:
        pyplot.show()
    pyplot.close()
