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
course_with_houses = pandas.concat([numerical_features, dataset["Hogwarts House"]], axis=1)

ravenclaw = course_with_houses.loc[course_with_houses["Hogwarts House"] == "Ravenclaw"]
slytherin = course_with_houses.loc[course_with_houses["Hogwarts House"] == "Slytherin"]
hufflepuff = course_with_houses.loc[course_with_houses["Hogwarts House"] == "Hufflepuff"]
gryffindor = course_with_houses.loc[course_with_houses["Hogwarts House"] == "Gryffindor"]

# Get a list of course pairs (Astro, Herbo)
courses_list = list(numerical_features.items())
courses_pairs = []

for course in courses_list:
    # Get list of all course except of index i
    other_courses = list(filter(lambda x: x != course, courses_list))
    # tuple of course[i], all other courses
    app = (course, other_courses)
    courses_pairs.append(app)

for given_course, given_courses in courses_pairs:
    for course in given_courses:
        print(f"{given_course[0]} - {course[0]}")
    print("\n")

# TODO scatter plot the pairs of classes
exit(0)
for label, _ in numerical_features.items():

    pyplot.style.use('gruvbox.mplstyle')
    pyplot.title(str(label))

    # pyplot.hist(ravenclaw[label], alpha=0.5, label="Ravenclaw")
    # pyplot.hist(slytherin[label], alpha=0.5, label="Slytherin")
    # pyplot.hist(hufflepuff[label], alpha=0.5, label="Hufflepuff")
    # pyplot.hist(gryffindor[label], alpha=0.5, label="Gryffindor")

    # pyplot.legend(loc='best')

    filename = f'static/Image/scatter/{label}.png'
    # pyplot.savefig(filename, format="png")
    # print(f'created {filename}')

    if show_plot == True:
        pyplot.show()
    pyplot.close()
