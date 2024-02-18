import argparse

import matplotlib.pyplot as pyplot
import seaborn

from data_preprocessing import create_dataframe
from variables import colors, labels_column

parser = argparse.ArgumentParser( description="A simple python program to print the pait plot of a given csv dataset")
parser.add_argument("filename", help="the dataset csv file")
parser.add_argument( "--show", action="store_true", help="hangs program to display plots")
args = parser.parse_args()


pyplot.style.use("gruvbox.mplstyle")
seaborn.set_theme(style="dark")

dataset = create_dataframe(args.filename)
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

if args.show is True:
    pyplot.show()
pyplot.close()
