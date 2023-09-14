#!/usr/bin/python3

import argparse
import pandas
import matplotlib.pyplot as pyplot
from handle_data import create_dataframe, split_by_houses, blue, green, yellow, red

pyplot.style.use('gruvbox.mplstyle')

parser = argparse.ArgumentParser(description="A simple python program to print the scatter plots of a given csv dataset")
parser.add_argument('filename', help='the dataset csv file')
parser.add_argument('--show', action='store_true', help='hangs program to display plots')
args = parser.parse_args()

dataset = create_dataframe(args.filename)


gryffindor, hufflepuff, ravenclaw, slytherin = split_by_houses(dataset)
gryffindor = gryffindor.select_dtypes(include=["float64"])
hufflepuff = hufflepuff.select_dtypes(include=["float64"])
ravenclaw = ravenclaw.select_dtypes(include=["float64"])
slytherin = slytherin.select_dtypes(include=["float64"])

subjects_list = [label for label, _ in gryffindor.items()]
# Get a list of course pairs (Astro, Herbo)
subjects_pairs = []

for subject in subjects_list:
    # Get list of all subject except of index i
    other_subjects = list(filter(lambda x: x != subject, subjects_list))
    # tuple of subject[i], all other subjects
    app = (subject, other_subjects)
    subjects_pairs.append(app)

for given_subject, other_subjects in subjects_pairs:
    for other_subject in other_subjects:
        title = f"{given_subject} - {other_subject}" 

        pyplot.title(title)

        # set xylabels
        pyplot.scatter(ravenclaw[given_subject], ravenclaw[other_subject], c=blue, alpha=0.6, label="Ravenclaw")
        pyplot.scatter(slytherin[given_subject], slytherin[other_subject], c=green, alpha=0.6, label="Slytherin")
        pyplot.scatter(hufflepuff[given_subject], hufflepuff[other_subject], c=yellow, alpha=0.6, label="Hufflepuff")
        pyplot.scatter(gryffindor[given_subject], gryffindor[other_subject], c=red, alpha=0.6, label="Gryffindor")
        pyplot.legend(loc='best')

        filename = f'static/Image/scatter/{title}.png'
        pyplot.savefig(filename, format="png")
        print(f'created {filename}')

        if args.show == True:
            pyplot.show()
        pyplot.close()
