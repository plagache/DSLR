#!/usr/bin/python3

import seaborn
import argparse
import matplotlib.pyplot as pyplot
from handle_data import create_dataframe, blue, green, yellow, red

color_palette = {
        "Gryffindor": red,
        "Slytherin": green,
        "Hufflepuff": yellow,
        "Ravenclaw": blue
        }

seaborn.set_theme(style="ticks")

parser = argparse.ArgumentParser(description="A simple python program to print the pait plot of a given csv dataset")
parser.add_argument('filename', help='the dataset csv file')
parser.add_argument('--show', action='store_true', help='hangs program to display plots')
args = parser.parse_args()

dataset = create_dataframe(args.filename)

subjects = [
        'Arithmancy',
        'Astronomy',
        'Herbology',
        'Defense Against the Dark Arts',
        'Divination',
        'Muggle Studies',
        'Ancient Runes',
        'History of Magic',
        'Transfiguration',
        'Potions',
        'Care of Magical Creatures',
        'Charms',
        'Flying'
        ]
splot = seaborn.pairplot(dataset, vars=subjects, hue="Hogwarts House", palette=color_palette, diag_kind="hist")

filename = 'static/Image/pair/pairplot.png'
splot.savefig(filename)
print(f'created {filename}')

if args.show == True:
    pyplot.show()
pyplot.close()
