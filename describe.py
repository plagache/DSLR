#!/usr/bin/python3


import sys
import numpy
import math

from handle_data import create_dataframe

print(sys.argv)

nbr_arg = len(sys.argv)

if nbr_arg >= 2:
    print(sys.argv[1])
    dataset = create_dataframe(sys.argv[1])


def print_dataset():
    # print("\nDataset :\n", dataset)
    # print("\nDataset from index 2 to 9:\n", dataset.loc[2:9])
    # print("\nDataset column 0:\n", dataset.iloc[:,0])
    # print("\nDataset column 2 with name:\n", dataset.loc[:,"Hogwarts House"])
    # print("\nDataset column 2 with index:\n", dataset.iloc[:,1])
    # print("\nDataset column 2 with name:\n", dataset.loc[:,"Arithmancy"])
    # print("\nDataset column 2 with index:\n", dataset.iloc[:,7])
    print("\nDataset column 2 with name:\n", dataset.loc[:,"Herbology"])
    print("\nDataset column 2 with index:\n", dataset.iloc[:,8])
    print("\nColumn label of index 1:\n", dataset.columns[1])
    print("\nDataset columns label:\n", dataset.columns)

print_dataset()


def ft_count(array):
    counter = 0
    for element in array:
        if element == element:
            counter += 1
    return counter

count = ft_count(dataset.iloc[:,8])
print("my count: ", count)
print("count:", dataset.iloc[:,8].count())


def ft_mean(array):
    total : float = 0
    for element in array:
        # if isinstance(element, (int, float)) == True:
        if element == element:
            total += element
    return total / ft_count(array)


mean = ft_mean(dataset.iloc[:,8])
print("my mean:", mean)
print("mean:", dataset.iloc[:,8].mean())
# mean = ft_mean(dataset.iloc[:,7])
# print("my mean: ", mean)
# mean = ft_mean(dataset.iloc[:,6])
# print("my mean: ", mean)


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

std = standard_deviation(dataset.iloc[:,8])

print("my standard_deviation:", std)

print("standard_deviation:", numpy.std(dataset.iloc[:,8]))


def minimum(array):
    minimum = 0
    for element in array:
        if element == element:
            if element < minimum:
                minimum = element
    return minimum

min = minimum(dataset.iloc[:,8])

print("my minimum:", min)

print("minimum:", dataset.iloc[:,8].min())


def maximum(array):
    maximum = 0
    for element in array:
        if element == element:
            if element > maximum:
                maximum = element
    return maximum

max = maximum(dataset.iloc[:,8])

print("my maximum:", max)

print("maximum:", dataset.iloc[:,8].max())


def percentile(array, percent : float):
    sorted_array = array.sort_values(ignore_index=True)
    # print("\nsorted array:", sorted_array)
    value = 0
    count = ft_count(dataset.iloc[:,8])
    index = count * percent
    floor = math.floor(index)
    ceil = math.ceil(index)
    # print("\nindex:", index)
    # print("\nfloor:", floor)
    # print("\nceil:", ceil)

    floor_value = sorted_array.loc[floor]
    # print("\nfloor value:", floor_value)
    ceil_value = sorted_array.loc[ceil]
    # print("\nceil value:", ceil_value)

    value = floor_value * percent + ceil_value * (1 - percent)
    # for element in array:
    #     if element == element:
    
    # -4.308182
    # percent = 0
    return value

my25 = percentile(dataset.iloc[:,8], 0.25)
print("my 25%:", my25)

my50 = percentile(dataset.iloc[:,8], 0.5)
print("my 50%:", my50)

my75 = percentile(dataset.iloc[:,8], 0.75)
print("my 75%:", my75)

print(dataset.iloc[:,8].describe())
