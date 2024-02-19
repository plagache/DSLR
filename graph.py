import argparse

import matplotlib.pyplot as plt
import numpy as np

from data_preprocessing import create_dataframe
from variables import colors


def draw_losses(losses, classes):
    total = len(losses)

    for class_name, content in losses.items():
        plt.plot(np.linspace(0, total, total), content, ".", c=colors[class_name])
        plt.xlabel("step")
        plt.ylabel("loss")
    plt.legend(classes)
    plt.savefig("static/Image/loss/losses.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    print("\n------------ Graph -----------")

    parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
    parser.add_argument("filename", help="the loss csv file")
    args = parser.parse_args()

    dataset = create_dataframe(args.filename)

    plt.style.use("gruvbox.mplstyle")

    classes = dataset.columns.tolist()
    plt = draw_losses(dataset, classes)
