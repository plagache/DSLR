import argparse

import pandas

from data_preprocessing import (cleanup_nan, create_classes, create_dataframe,
                                split_by_classes)
from f_statistics import (ft_count, ft_mean, maximum, minimum, percentile,
                          standard_deviation)

parser = argparse.ArgumentParser(
    description="A simple python program to print a summary of a given csv dataset"
)
parser.add_argument("filename", help="the dataset csv file")
parser.add_argument("--web", action="store_true", help="export data to html output")
args = parser.parse_args()

dataset = create_dataframe(args.filename)

numerical_features = dataset.select_dtypes(include=["float64"])

cleaned = cleanup_nan(numerical_features)


def dataset_to_dic(dataset):
    dictionnaire = {
        "name": dataset.name,
        "count": ft_count(dataset),
        "mean": ft_mean(dataset),
        "standard deviation": standard_deviation(dataset),
        "min": minimum(dataset),
        "first": percentile(dataset, 0.25),
        "second": percentile(dataset, 0.5),
        "third": percentile(dataset, 0.75),
        "max": maximum(dataset),
    }
    return dictionnaire


def getDescribeDataframe(cleaned_df):
    column_list = []
    for _, serie in cleaned_df.items():
        column_list.append(dataset_to_dic(serie))

    described_df = pandas.DataFrame(column_list)
    described_transposed = described_df.transpose()

    header = described_transposed.iloc[0]
    described_transposed = described_transposed[1:]
    described_transposed.columns = header
    return described_transposed


described_df = getDescribeDataframe(cleaned)


def writeToHtmlTable(dataframe, table_name=""):
    filename = "templates/describe_table.html"
    if table_name != "":
        filename = "templates/describe_table_{}.html".format(table_name)

    with open(filename, "w") as table_html:
        table_html.write("<html>")
        table_html.write(dataframe.to_html())
        table_html.write("</html>")


if args.web is True:
    writeToHtmlTable(described_df)

    datasets = split_by_classes(dataset)
    classes = create_classes(dataset)
    for dataset, class_name in zip(datasets, classes):
        dataset = cleanup_nan(dataset.select_dtypes(include=["float64"]))
        described_dataset = getDescribeDataframe(dataset)
        writeToHtmlTable(described_dataset, class_name)
else:
    print(described_df.to_string())
