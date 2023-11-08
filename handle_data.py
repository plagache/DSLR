#!/usr/bin/python3

import pandas
import math

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
    # dataset = dataset.dropna()
    print("-------------", dataset, "-----------------")
    numerical_features = dataset.select_dtypes(include=["float64"])
    houses = ["Gryffindor", "Ravenclaw", "Slytherin", "Hufflepuff"]
    houses.remove(house)
    houses.insert(0, house)
    dataset = dataset.replace(houses, [1., 0., 0., 0.])
    classer = dataset["Hogwarts House"]
    return classer, numerical_features

def ft_count(array):
    counter = 0
    for element in array:
        if element == element:
            counter += 1
    return counter


def ft_mean(array):
    total : float = 0
    for element in array:
        # if isinstance(element, (int, float)) == True:
        if element == element:
            total += element
    return total / ft_count(array)



def standard_deviation(array):

    count = ft_count(array)
    mean = ft_mean(array)
    variance = 0
    deviation = 0
    standard_deviation = 0

    for element in array:
        if element == element:
            deviation += (element - mean)**2

    variance = deviation / count

    standard_deviation = math.sqrt(variance)

    return standard_deviation


def minimum(array):
    minimum = 0
    for element in array:
        if element == element:
            if element < minimum:
                minimum = element
    return minimum


def maximum(array):
    maximum = 0
    for element in array:
        if element == element:
            if element > maximum:
                maximum = element
    return maximum


def percentile(array, percent : float):
    sorted_array = array.sort_values(ignore_index=True)
    # print("\nsorted array:", sorted_array)
    value = 0
    count = ft_count(array)
    # print(count)
    # méthode recommandée par le National Institute of Standards and Technology (NIST)
    # 1 + P*(n-1)/100
    rank = 1 + percent * (count - 1)
    index = rank - 1
    # index = count * percent
    floor = math.floor(index)
    ceil = math.ceil(index)
    # print("\nindex:", index)
    # print("\nfloor:", floor)
    # print("\nceil:", ceil)
    reste_floor = index - floor
    reste_ceil = ceil - index
    # print("\nreste floor:", reste_floor)
    # print("\nreste ceil:", reste_ceil)

    floor_value = sorted_array.loc[floor]
    # print("\nfloor value:", floor_value)
    ceil_value = sorted_array.loc[ceil]
    # print("\nceil value:", ceil_value)

    if index - floor != 0:
        value = floor_value + reste_floor * (ceil_value - floor_value)
        # print("value new :", value)
        # value = (floor_value * reste_ceil) + (ceil_value * reste_floor)
    else:
        # index_value = sorted_array.loc[index - 1]
        # print(index_value)
        value = sorted_array.loc[index]
        # index_value = sorted_array.loc[index + 1]
        # print(index_value)
    # for element in array:
    #     if element == element:
    
    # -4.308182
    # percent = 0
    # return value
    return round(value, 6)
