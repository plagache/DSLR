#!/usr/bin/python3

import sys
import pandas
import getopt

from handle_data import create_dataframe
import matplotlib.pyplot as pyplot

options, args = getopt.getopt(sys.argv[1:], '', ['show'])
nbr_arg = len(args)
nbr_opt = len(options)

show_plot = False
if nbr_opt > 0:
    for option in options:
        if option[0] == '--show':
            show_plot = True

dataset = pandas.DataFrame()
if nbr_arg >= 1:
    dataset = create_dataframe(args[0])
else:
    print("no dataset provided\nusage...")
    exit(1)

numerical_features = dataset.select_dtypes(include=["float64"])
subject_with_houses = pandas.concat([numerical_features, dataset["Hogwarts House"]], axis=1)

ravenclaw = subject_with_houses.loc[subject_with_houses["Hogwarts House"] == "Ravenclaw"]
slytherin = subject_with_houses.loc[subject_with_houses["Hogwarts House"] == "Slytherin"]
hufflepuff = subject_with_houses.loc[subject_with_houses["Hogwarts House"] == "Hufflepuff"]
gryffindor = subject_with_houses.loc[subject_with_houses["Hogwarts House"] == "Gryffindor"]

# Get a list of course pairs (Astro, Herbo)
subjects_list = [label for label, _ in numerical_features.items()]
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

        pyplot.style.use('gruvbox.mplstyle')
        pyplot.title(title)

        # set xylabels
        pyplot.scatter(ravenclaw[given_subject], ravenclaw[other_subject], alpha=0.8, label="Ravenclaw")
        pyplot.scatter(slytherin[given_subject], slytherin[other_subject], alpha=0.8, label="Slytherin")
        pyplot.scatter(hufflepuff[given_subject], hufflepuff[other_subject], alpha=0.8, label="Hufflepuff")
        pyplot.scatter(gryffindor[given_subject], gryffindor[other_subject], alpha=0.8, label="Gryffindor")
        pyplot.legend(loc='best')

        filename = f'static/Image/scatter/{title}.png'
        pyplot.savefig(filename, format="png")
        print(f'created {filename}')

        if show_plot == True:
            pyplot.show()
        pyplot.close()
