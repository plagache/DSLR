import math


def ft_count(array):
    counter = 0
    for element in array:
        if element == element:
            counter += 1
    return counter


def ft_mean(array):
    total: float = 0
    for element in array:
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
            deviation += (element - mean) ** 2

    variance = deviation / count

    # standard_deviation = math.sqrt(variance)
    standard_deviation = variance**0.5

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


def percentile(array, percent: float):
    """
    méthode recommandée par le National Institute of Standards and Technology (NIST)
    1 + P*(n-1)/100
    """
    sorted_array = array.sort_values(ignore_index=True)
    value = 0
    count = ft_count(array)
    rank = 1 + percent * (count - 1)
    index = rank - 1
    floor = math.floor(index)
    ceil = math.ceil(index)
    reste_floor = index - floor
    # reste_ceil = ceil - index

    floor_value = sorted_array.loc[floor]
    ceil_value = sorted_array.loc[ceil]

    if index - floor != 0:
        value = floor_value + reste_floor * (ceil_value - floor_value)
    else:
        value = sorted_array.loc[index]

    return value
