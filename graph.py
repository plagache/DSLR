from data_preprocessing import create_dataframe
import numpy as np
import matplotlib.pyplot as plt
import argparse
from variables import colors

parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
parser.add_argument('filename', help='the loss csv file')
args = parser.parse_args()

dataset = create_dataframe(args.filename)

plt.style.use('gruvbox.mplstyle')


def draw_losses(losses, class_name):

    total = len(losses)

    plt.plot(np.linspace(0, total, total), losses, '.', c=colors[class_name])
    plt.title(class_name)
    plt.xlabel('step')
    plt.ylabel('loss')
    plt.savefig(f'static/Image/loss/{class_name}.png')
    plt.close()


for class_name, content in dataset.items():
    draw_losses(content, class_name)
