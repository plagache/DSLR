from data_preprocessing import create_dataframe
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
parser.add_argument('filename', help='the loss csv file')
args = parser.parse_args()

dataset = create_dataframe(args.filename)


def draw_losses(losses, house):

    total = len(losses)

    plt.plot(np.linspace(0, total, total), losses, 'g', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('step')
    plt.ylabel('Losses')
    plt.legend()
    # plt.show()
    plt.savefig(f'static/Image/loss/{house}.png')
    plt.close()


# print(dataset)

for label, content in dataset.items():
    draw_losses(content, label)
