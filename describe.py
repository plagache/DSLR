#!/usr/bin/python3


import sys
import numpy
import math
import pandas

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
    # print("\nDataset column 2 with name:\n", dataset.loc[:,"Herbology"])
    # print("\nDataset column 2 with index:\n", dataset.iloc[:,8])
    # print("\nColumn label of index 1:\n", dataset.columns[1])
    # print("\nDataset columns label:\n", dataset.columns)
    # print("\nDataset info:\n", dataset.info)
    # print("\nDataset len:\n", len(dataset.columns))
    print("\nDataset :\n", dataset.dtypes)

# print_dataset()

herbology = dataset.loc[:,"Herbology"]
column_8 = dataset.iloc[:,8]
# print(herbology)
# print(column_8)

numerical_features = dataset.select_dtypes(include=[numpy.float64])

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

count = ft_count(herbology)
print("my count: ", count)
print("count:", herbology.count())


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

print("de minimum:", dataset.iloc[:,8].min())


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

# weigth of the index / last rank should be reajust

my25 = percentile(dataset.iloc[:,8], 0.2500)
print("my 25%:", my25)

my50 = percentile(dataset.iloc[:,8], 0.5)
print("my 50%:", my50)

my75 = percentile(dataset.iloc[:,8], 0.75)
print("my 75%:", my75)

# print(dataset.iloc[:,8].describe())
print(dataset.iloc[:,8].quantile([0.25, 0.5, 0.75]))


my25 = percentile(dataset.iloc[:,7], 0.25)
print("my 25%:", my25)

my50 = percentile(dataset.iloc[:,7], 0.5)
print("my 50%:", my50)

my75 = percentile(dataset.iloc[:,7], 0.75)
print("my 75%:", my75)


# print(dataset.iloc[:,7].describe())
print(dataset.iloc[:,7].quantile([0.25, 0.5, 0.75]))
