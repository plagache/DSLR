import pandas
from f_statistics import percentile


def create_dataframe(csv_string):
    print(csv_string)
    csv_dataset = pandas.read_csv(csv_string)
    dataset = pandas.DataFrame(csv_dataset)
    dataset.sort_index(axis=1, inplace=True)
    return dataset


def cleanup_nan(dataframe):
    cleaned_series = []
    for _, serie in dataframe.items():
        serie.dropna(inplace=True, ignore_index=True)
        cleaned_series.append(serie)
    return pandas.concat(cleaned_series, axis=1)


def split_by_houses(dataframe):
    ravenclaw = dataframe.loc[dataframe["Hogwarts House"] == "Ravenclaw"]
    slytherin = dataframe.loc[dataframe["Hogwarts House"] == "Slytherin"]
    hufflepuff = dataframe.loc[dataframe["Hogwarts House"] == "Hufflepuff"]
    gryffindor = dataframe.loc[dataframe["Hogwarts House"] == "Gryffindor"]
    return gryffindor, hufflepuff, ravenclaw, slytherin


def create_training_data(dataset):
    # dataset = numerization(dataset)
    numerical_features = dataset.select_dtypes(include=["float64"])
    quartiles = set_quartiles(numerical_features)
    rescaled = robust_scale(numerical_features, quartiles)
    return rescaled


def create_classer(dataset, houses):

    classer = pandas.DataFrame()
    for house in houses:
        classer[house] = dataset["Hogwarts House"].map(lambda x: 1.0 if x == house else 0.0 )
    return classer


def numerization(dataset):
    dataset["Birthday"] = pandas.to_datetime(dataset["Birthday"]).astype(int)
    numerized_dataset = dataset.replace(regex={'Right': 1, 'Left': -1})
    return numerized_dataset


def set_quartiles(dataframe):
    quartiles = []

    for _, data in dataframe.items():
        first = percentile(data, 0.25)
        second = percentile(data, 0.5)
        third = percentile(data, 0.75)
        # First replace nan with median value
        # Then scale the values
        quartiles.append((data.name, first, second, third))

    df = pandas.DataFrame(quartiles, columns=["Courses", "Q1", "Q2", "Q3"])
    df.to_csv("tmp/quartiles.csv", index=False)

    return quartiles


def robust_scale(dataframe: pandas.DataFrame, quartiles):
    ret = pandas.DataFrame()
    for (name, data), (courses, first, second, third) in zip(dataframe.items(), quartiles):
        # First replace nan with median value
        # Then scale the values
        # print(name, courses)
        ret[name] = data.fillna(second).map(lambda x: (x - second) / (third - first))
    return ret
