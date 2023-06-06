#!/usr/bin/python3


import sys
import numpy

from handle_data import create_dataframe

print(sys.argv)

nbr_arg = len(sys.argv)

if nbr_arg >= 2:
    print(sys.argv[1])
    dataset = create_dataframe(sys.argv[1])


def print_dataset():
    # print("\nDataset :\n", dataset)
    print("\nDataset from index 2 to 9:\n", dataset.loc[2:9])
    print("\nDataset column 0:\n", dataset.iloc[:,0])
    print("\nDataset column 2 with name:\n", dataset.loc[:,"Hogwarts House"])
    print("\nDataset column 2 with index:\n", dataset.iloc[:,1])
    print("\nColumn label of index 1:\n", dataset.columns[1])
    print("\nDataset columns label:\n", dataset.columns)

# print_dataset()



def ft_count(array):
    counter = 0
    for element in array:
        counter += 1
    return counter

count = ft_count(dataset.iloc[:,7])
print(count)

def ft_mean(array):
    print (array)
    total : float = 0
    print(total)
    for element in array:
        print(total)
        # print(element)
        # print(type(element))
        # print(isinstance(element, (int, float)))
        if isinstance(element, (int, float)) == True:
        # if type(element) == 'float':
            total += element
            # print("element", element)
            # print("total\n", total)
        # print("total\n", total)
    return total / ft_count(array)




mean = ft_mean(dataset.iloc[:,7])
mean = ft_mean(dataset.iloc[:,6])

print(mean)
