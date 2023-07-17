#!/usr/bin/python3

import sys
import numpy
import math
import pandas
import getopt

from handle_data import create_dataframe

options, args = getopt.getopt(sys.argv[1:], '', ['web'])
nbr_arg = len(args)
nbr_opt = len(options)

html_ouput = False
if nbr_opt > 0:
    for option in options:
        if option[0] == '--web':
            html_ouput = True

dataset = pandas.DataFrame()
if nbr_arg >= 1:
    dataset = create_dataframe(args[0])
else:
    print("no dataset provided\nusage...")
    exit(1)


def print_dataset():
    print("\nDataset :\n", dataset)
    # print("\nDataset from index 2 to 9:\n", dataset.loc[2:9])
    # print("\nDataset column 0:\n", dataset.iloc[:,0])
    # print("\nDataset column 2 with name:\n", dataset.loc[:,"Hogwarts House"])
    # print("\nDataset column 2 with index:\n", dataset.iloc[:,1])
    # print("\nDataset column 2 with name:\n", dataset.loc[:,"Arithmancy"])
    # print("\nDataset column 2 with index:\n", dataset.iloc[:,7])
    # print("\nDataset column 2 with name:\n", dataset.loc[:,"Herbology"])
    # print("\nDataset column 2 with index:\n", dataset.iloc[:,8])
    # print("\nColumn label of index 1:\n", dataset.columns[1])
    # print("\nDataset columns label:\n", dataset.columns)
    # print("\nDataset info:\n", dataset.info)
    # print("\nDataset len:\n", len(dataset.columns))
    print("\nDataset :\n", dataset.dtypes)

# print_dataset()

numerical_features = dataset.select_dtypes(include=["float64"])

def cleanup(dataframe):
    cleaned_series = []
    for _, serie in dataframe.items():
        serie.dropna(inplace=True, ignore_index=True)
        cleaned_series.append(serie)
    return pandas.concat(cleaned_series, axis=1)

cleaned = cleanup(numerical_features)
# print(cleaned)


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


def dataset_to_dic(dataset):
    dictionnaire = {
        "name" : dataset.name,
        "count" : ft_count(dataset),
        "mean" : ft_mean(dataset),
        "standard deviation" : standard_deviation(dataset),
        "min" : minimum(dataset),
        "first" : percentile(dataset, 0.25),
        "second" : percentile(dataset, 0.5),
        "third" : percentile(dataset, 0.75),
        "max" : maximum(dataset),
                    }
    return dictionnaire


dic_list = []
for _, serie in cleaned.items():
    dic_list.append(dataset_to_dic(serie))

new = pandas.DataFrame(dic_list)
new_transpose = new.transpose()

header=new_transpose.iloc[0]
new_transpose=new_transpose[1:]
new_transpose.columns=header
# print(new_transpose)
# exit(0)
if html_ouput == True:
    with open("templates/describe_table.html", "w") as table_html:
        table_html.write("<html>")
        table_html.write(new_transpose.to_html())
        table_html.write("</html>")
else:
    print(new_transpose.to_string())
