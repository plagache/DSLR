#!/usr/bin/python3

import pandas
import seaborn
import argparse
import matplotlib.pyplot as pyplot
from handle_data import create_dataframe, blue, green, yellow, red

color_palette = {
        "Gryffindor": red,
        "Slytherin": green,
        "Hufflepuf": yellow,
        "Ravenclaw": blue
        }

seaborn.set_theme(style="ticks")

parser = argparse.ArgumentParser(description="A simple python program to print the pait plot of a given csv dataset")
parser.add_argument('filename', help='the dataset csv file')
parser.add_argument('--show', action='store_true', help='hangs program to display plots')
args = parser.parse_args()

dataset = create_dataframe(args.filename)

dataset.drop(columns=['First Name','Last Name','Birthday','Best Hand'])

print("PLOT")
splot = seaborn.pairplot(dataset, hue="Hogwarts House", palette=color_palette, diag_kind="hist")

print("SAVEFIG")
filename = 'static/Image/pair/pairplot.png'
splot.savefig(filename)

print(f'created {filename}')
# pyplot.show()
print("CLOSE")
pyplot.close()

if args.show == True:
    print("HERE SHOW TRUE")
    exit(0)
