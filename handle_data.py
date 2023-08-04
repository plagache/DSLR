#!/usr/bin/python3

import pandas


def create_dataframe(csv_string):
    print(csv_string)
    csv_dataset = pandas.read_csv(csv_string)
    dataset = pandas.DataFrame(csv_dataset)
    dataset.sort_index(axis=1, inplace=True)

    return dataset
