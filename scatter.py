#!/usr/bin/python3

import sys
import numpy
import math
import pandas

from handle_data import create_dataframe
import matplotlib.pyplot as pyplot

print(sys.argv)

nbr_arg = len(sys.argv)

dataset = pandas.DataFrame()
if nbr_arg >= 2:
    print(sys.argv[1])
    dataset = create_dataframe(sys.argv[1])

numerical_features = dataset.select_dtypes(include=[numpy.float64])
course_with_houses = pandas.concat([numerical_features, dataset["Hogwarts House"]], axis=1)
sorted = course_with_houses.sort_values("Hogwarts House")
# print(sorted)

for label, serie in sorted.items():
    if label != "Hogwarts House":
        pyplot.style.use('gruvbox.mplstyle')
        # pyplot.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-dark.mplstyle')
        ravenclaw = sorted.loc[sorted["Hogwarts House"] == "Ravenclaw",label]
        slytherin = sorted.loc[sorted["Hogwarts House"] == "Slytherin",label]
        hufflepuff = sorted.loc[sorted["Hogwarts House"] == "Hufflepuff",label]
        gryffindor = sorted.loc[sorted["Hogwarts House"] == "Gryffindor",label]
        # sorted.plot.hist(column=label ,by="Hogwarts House", label=label, sharex=True, sharey=True)
        pyplot.hist(ravenclaw, alpha=0.5, label=label)
        pyplot.hist(slytherin, alpha=0.5, label=label)
        pyplot.hist(hufflepuff, alpha=0.5, label=label)
        pyplot.hist(gryffindor, alpha=0.5, label=label)
    pyplot.legend(loc='best')

    # filename = f'ressources/{label}.png'
    filename = f'static/Image/hist.png'
    pyplot.savefig(filename, format="png")

    pyplot.show()
    break
