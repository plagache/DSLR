#!/usr/bin/python3

import pandas

blue = '#83a598'
green = '#b8bb26'
yellow = '#fabd2f'
red = '#fb4934'

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

def classer(dataset, house):
    dataset = dataset.dropna()
    print("-------------", dataset, "-----------------")
    numerical_features = dataset.select_dtypes(include=["float64"])
    houses = ["Gryffindor", "Ravenclaw", "Slytherin", "Hufflepuff"]
    houses.remove(house)
    houses.insert(0, house)
    dataset = dataset.replace(houses, [1., 0., 0., 0.])
    classer = dataset["Hogwarts House"]
    return classer, numerical_features
