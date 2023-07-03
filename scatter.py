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
sorted = course_with_houses.sort_values("Hogwarts House")
# print(sorted)

ravenclaw = sorted.loc[sorted["Hogwarts House"] == "Ravenclaw"]
slytherin = sorted.loc[sorted["Hogwarts House"] == "Slytherin"]
hufflepuff = sorted.loc[sorted["Hogwarts House"] == "Hufflepuff"]
gryffindor = sorted.loc[sorted["Hogwarts House"] == "Gryffindor"]
for label, serie in sorted.items():
    if label != "Hogwarts House":
        pyplot.style.use('gruvbox.mplstyle')
        pyplot.hist(ravenclaw[label], alpha=0.5, label="Ravenclaw")
        pyplot.hist(slytherin[label], alpha=0.5, label="Slytherin")
        pyplot.hist(hufflepuff[label], alpha=0.5, label="Hufflepuff")
        pyplot.hist(gryffindor[label], alpha=0.5, label="Gryffindor")
    pyplot.legend(loc='best')

    # filename = f'ressources/{label}.png'
    filename = f'static/Image/hist.png'
    pyplot.savefig(filename, format="png")

    pyplot.show()
    break
