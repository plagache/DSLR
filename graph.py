import argparse

import matplotlib.pyplot as plt
import numpy as np

from data_preprocessing import create_dataframe
from variables import colors, learning_rate, stochastic


def draw_graphs(losses, classes, accuracies):
    total = len(losses)

    plt.style.use("gruvbox.mplstyle")

    figure, axes = plt.subplots()

    for class_name, content in losses.items():
        axes.plot(np.linspace(0, total, total), content, ".", c=colors[class_name])

    axes.plot(np.linspace(0, total, total), accuracies, ".", c="#ebdbb2")

    axes.set_xlabel("step")
    axes.set_ylabel("loss")
    axes.legend(classes)

    return figure


if __name__ == "__main__":
    print("\n------------ Graph -----------")

    parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
    parser.add_argument("losses", help="the loss csv file")
    parser.add_argument("accuracies", help="the loss csv file")
    args = parser.parse_args()

    losses = create_dataframe(args.losses)
    accuracies = create_dataframe(args.accuracies)

    classes = losses.columns.tolist()

    figure = draw_graphs(losses, classes, accuracies)

    figure.suptitle(f"learning rate: {learning_rate} | Stochastic: {stochastic}")
    figure.savefig(f"static/Image/loss/lr_{learning_rate}_st_{stochastic}.png", dpi=200)
    plt.show()
    plt.close()
