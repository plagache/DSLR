#!/usr/bin/python3

import sys
import pandas

from handle_data import create_dataframe
import matplotlib.pyplot as pyplot

print(sys.argv)

nbr_arg = len(sys.argv)

dataset = pandas.DataFrame()
if nbr_arg >= 2:
    print(sys.argv[1])
    dataset = create_dataframe(sys.argv[1])

numerical_features = dataset.select_dtypes(include=["float64"])
course_with_houses = pandas.concat([numerical_features, dataset["Hogwarts House"]], axis=1)

ravenclaw = course_with_houses.loc[course_with_houses["Hogwarts House"] == "Ravenclaw"]
slytherin = course_with_houses.loc[course_with_houses["Hogwarts House"] == "Slytherin"]
hufflepuff = course_with_houses.loc[course_with_houses["Hogwarts House"] == "Hufflepuff"]
gryffindor = course_with_houses.loc[course_with_houses["Hogwarts House"] == "Gryffindor"]

for label, _ in numerical_features.items():

    pyplot.style.use('gruvbox.mplstyle')
    pyplot.title(str(label))

    pyplot.hist(ravenclaw[label], alpha=0.5, label="Ravenclaw")
    pyplot.hist(slytherin[label], alpha=0.5, label="Slytherin")
    pyplot.hist(hufflepuff[label], alpha=0.5, label="Hufflepuff")
    pyplot.hist(gryffindor[label], alpha=0.5, label="Gryffindor")

    pyplot.legend(loc='best')

    filename = f'static/Image/hist/{label}.png'
    pyplot.savefig(filename, format="png")
    print(f'created {filename}.png')

    # pyplot.show()
    pyplot.close()
