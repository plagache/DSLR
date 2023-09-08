#!/usr/bin/python3

import pandas
import argparse
import matplotlib.pyplot as pyplot
from handle_data import create_dataframe, split_by_houses

parser = argparse.ArgumentParser(description="A simple python program to print the pait plot of a given csv dataset")
parser.add_argument('filename', help='the dataset csv file')
parser.add_argument('--show', action='store_true', help='hangs program to display plots')
args = parser.parse_args()

dataset = create_dataframe(args.filename)

gryffindor, hufflepuff, ravenclaw, slytherin = split_by_houses(dataset)
gryffindor = gryffindor.select_dtypes(include=["float64"])
hufflepuff = hufflepuff.select_dtypes(include=["float64"])
ravenclaw = ravenclaw.select_dtypes(include=["float64"])
slytherin = slytherin.select_dtypes(include=["float64"])
