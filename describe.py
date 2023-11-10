#!/usr/bin/python3

import pandas
import argparse
from handle_data import create_dataframe, cleanup_nan, split_by_houses, ft_count, ft_mean, standard_deviation, minimum, maximum, percentile

parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
parser.add_argument('filename', help='the dataset csv file')
parser.add_argument('--web', action='store_true', help='export data to html output')
args = parser.parse_args()

dataset = create_dataframe(args.filename)

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

cleaned = cleanup_nan(numerical_features)
# print(cleaned)


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

def getDescribeDataframe(cleaned_df):
    column_list = []
    for _, serie in cleaned_df.items():
        column_list.append(dataset_to_dic(serie))

    described_df = pandas.DataFrame(column_list)
    described_transposed = described_df.transpose()

    # save column name
    header = described_transposed.iloc[0]
    # cut column name from df
    described_transposed = described_transposed[1:]
    # reinscribed column name
    described_transposed.columns = header
    return described_transposed

described_df = getDescribeDataframe(cleaned)

def writeToHtmlTable(dataframe, table_name = ""):
    filename = "templates/describe_table.html"
    if table_name != "":
        filename = "templates/describe_table_{}.html".format(table_name)

    with open(filename, "w") as table_html:
        table_html.write("<html>")
        table_html.write(dataframe.to_html())
        table_html.write("</html>")

if args.web == True:
    writeToHtmlTable(described_df)

    gryffindor, hufflepuff, ravenclaw, slytherin = split_by_houses(dataset)

    gryffindor = cleanup_nan(gryffindor.select_dtypes(include=["float64"]))
    described_gryffindor = getDescribeDataframe(gryffindor)
    writeToHtmlTable(described_gryffindor, "gryffindor")

    hufflepuff = cleanup_nan(hufflepuff.select_dtypes(include=["float64"]))
    described_hufflepuff = getDescribeDataframe(hufflepuff)
    writeToHtmlTable(described_hufflepuff, "hufflepuff")

    ravenclaw = cleanup_nan(ravenclaw.select_dtypes(include=["float64"]))
    described_ravenclaw = getDescribeDataframe(ravenclaw)
    writeToHtmlTable(described_ravenclaw, "ravenclaw")

    slytherin = cleanup_nan(slytherin.select_dtypes(include=["float64"]))
    described_slytherin = getDescribeDataframe(slytherin)
    writeToHtmlTable(described_slytherin, "slytherin")
else:
    print(described_df.to_string())
