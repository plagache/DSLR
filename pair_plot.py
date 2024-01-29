import seaborn
import argparse
import matplotlib.pyplot as pyplot
from f_statistics import blue, green, yellow, red
from data_preprocessing import create_dataframe

parser = argparse.ArgumentParser(description="A simple python program to print the pait plot of a given csv dataset")
parser.add_argument('filename', help='the dataset csv file')
parser.add_argument('--show', action='store_true', help='hangs program to display plots')
args = parser.parse_args()

color_palette = {
        "Gryffindor": red,
        "Slytherin": green,
        "Hufflepuff": yellow,
        "Ravenclaw": blue
        }

pyplot.style.use('gruvbox.mplstyle')
seaborn.set_theme(style="dark")

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
subjects.sort()
splot = seaborn.pairplot(dataset, vars=subjects, hue="Hogwarts House", palette=color_palette, diag_kind="hist", plot_kws=dict(marker='.', alpha=0.8, sizes=5))

filename = 'static/Image/pair/pairplot.png'
splot.savefig(filename)
print(f'created {filename}')

if args.show is True:
    pyplot.show()
pyplot.close()
