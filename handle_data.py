#!/usr/bin/python3

import pandas

def create_dataframe(csv_string):
    print(csv_string)
    csv_dataset = pandas.read_csv(csv_string)
    dataset = pandas.DataFrame(csv_dataset)
    dataset.sort_index(axis=1, inplace=True)

    return dataset

def split_by_houses(dataframe):
    ravenclaw = dataframe.loc[dataframe["Hogwarts House"] == "Ravenclaw"]
    slytherin = dataframe.loc[dataframe["Hogwarts House"] == "Slytherin"]
    hufflepuff = dataframe.loc[dataframe["Hogwarts House"] == "Hufflepuff"]
    gryffindor = dataframe.loc[dataframe["Hogwarts House"] == "Gryffindor"]
    return gryffindor, hufflepuff, ravenclaw, slytherin
