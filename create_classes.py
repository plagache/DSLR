#!/usr/bin/python3

import pandas


def create_classe(csv_string):
    print(csv_string)
    csv_dataset = pandas.read_csv(csv_string)
    dt_dataset = pandas.DataFrame(csv_dataset)

    print(dt_dataset)

    class dataset:
        nbr = dt_dataset["Index"]
        house = dt_dataset["Hogwarts House"]


    # print(dataset.nbr)
    # print(dataset.house)
    return dataset



# create_classe()
