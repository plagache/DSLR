import argparse

import matplotlib.pyplot as plt
import numpy as np

from data_preprocessing import create_dataframe
from variables import colors, learning_rate, stochastic


def draw_graphs(losses, classes, accuracies):
    total = len(losses)

    for class_name, content in losses.items():
        plt.plot(np.linspace(0, total, total), content, ".", c=colors[class_name])
        plt.xlabel("step")
        plt.ylabel("loss")
    plt.plot(np.linspace(0, total, total), accuracies, ".", c="#ebdbb2")
    plt.legend(classes)
    plt.title(f"learning rate: {learning_rate} | Stochastic: {stochastic}")
    plt.savefig(f"static/Image/loss/lr_{learning_rate}_st_{stochastic}.png", dpi=200)
    plt.show()
    plt.close()


if __name__ == "__main__":
    print("\n------------ Graph -----------")

    parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
    parser.add_argument("losses", help="the loss csv file")
    parser.add_argument("accuracies", help="the loss csv file")
    args = parser.parse_args()

    losses = create_dataframe(args.losses)
    accuracies = create_dataframe(args.accuracies)

    plt.style.use("gruvbox.mplstyle")

    classes = losses.columns.tolist()
    # plt = draw_graphs(losses, classes)
    plt = draw_graphs(losses, classes, accuracies)
