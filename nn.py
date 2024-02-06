import numpy as np


class Neuron():

    def __init__(self, number_of_input, weight=None):
        self.weight = weight if weight is not None else np.zeros(number_of_input, dtype=np.float64)

    def outputs(self, tensor):
        self._z = self.weight @ tensor
        self._outputs = self.logistic(self._z)
        return self._outputs

    # This is an activation function of the sigmoid type
    # in our case a logistic one (it normalize our result between 0 and 1)
    # https://en.wikipedia.org/wiki/Logistic_function
    def logistic(self, z):
        return 1.0 / (1 + np.exp(-z))
