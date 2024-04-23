import argparse

import pandas

from data_preprocessing import cleanup_nan, create_classes, create_dataframe, split_by_classes
from f_statistics import ft_count, ft_mean, maximum, minimum, percentile, standard_deviation

parser = argparse.ArgumentParser(description="A simple python program to print a summary of a given csv dataset")
parser.add_argument("filename", help="the dataset csv file")
parser.add_argument("--web", action="store_true", help="export data to html output")
args = parser.parse_args()

dataset = create_dataframe(args.filename)

numerical_features = dataset.select_dtypes(include=["float64"])

cleaned = cleanup_nan(numerical_features)


def dataset_to_dic(dataset):
    dictionnaire = {
        "name": dataset.name,
        "Count": ft_count(dataset),
        "Mean": ft_mean(dataset),
        "Std": standard_deviation(dataset),
        "Min": minimum(dataset),
        "25%": percentile(dataset, 0.25),
        "50%": percentile(dataset, 0.5),
        "75%": percentile(dataset, 0.75),
        "Max": maximum(dataset),
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
described_df.rename_axis(None, axis="columns", inplace=True)


def writeToHtmlTable(dataframe, table_name=""):
    filename = "templates/describe_table.html"
    if table_name != "":
        filename = "templates/describe_table_{}.html".format(table_name)

    with open(filename, "w") as table_html:
        table_html.write("<html>")
        table_html.write(dataframe.to_html())
        table_html.write("</html>")


if args.web is True:
    described_df.rename_axis("All dataset", axis="columns", inplace=True)
    writeToHtmlTable(described_df)

    datasets = split_by_classes(dataset)
    classes = create_classes(dataset)
    for dataset, class_name in zip(datasets, classes):
        dataset = cleanup_nan(dataset.select_dtypes(include=["float64"]))
        described_dataset = getDescribeDataframe(dataset)
        described_dataset.rename_axis(class_name, axis="columns", inplace=True)
        writeToHtmlTable(described_dataset, class_name)
else:
    print(described_df.to_string())
