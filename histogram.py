import argparse
import matplotlib.pyplot as pyplot
from variables import blue, green, yellow, red
from data_preprocessing import create_dataframe, split_by_houses

parser = argparse.ArgumentParser(description="A simple python program to print the histogram plots of a given csv dataset")
parser.add_argument('filename', help='the dataset csv file')
parser.add_argument('--show', action='store_true', help='hangs program to display plots')
args = parser.parse_args()

pyplot.style.use('gruvbox.mplstyle')
dataset = create_dataframe(args.filename)


gryffindor, hufflepuff, ravenclaw, slytherin = split_by_houses(dataset)
gryffindor = gryffindor.select_dtypes(include=["float64"])
hufflepuff = hufflepuff.select_dtypes(include=["float64"])
ravenclaw = ravenclaw.select_dtypes(include=["float64"])
slytherin = slytherin.select_dtypes(include=["float64"])

for label, _ in gryffindor.items():

    pyplot.title(str(label))

    pyplot.hist(ravenclaw[label], color=blue, alpha=0.5, label="Ravenclaw")
    pyplot.hist(slytherin[label], color=green, alpha=0.5, label="Slytherin")
    pyplot.hist(hufflepuff[label], color=yellow, alpha=0.5, label="Hufflepuff")
    pyplot.hist(gryffindor[label], color=red, alpha=0.5, label="Gryffindor")

    pyplot.legend(loc='best')

    filename = f'static/Image/hist/{label}.png'
    pyplot.savefig(filename, format="png")
    print(f'created {filename}')

    if args.show is True:
        pyplot.show()
    pyplot.close()
