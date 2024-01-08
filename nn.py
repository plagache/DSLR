import numpy as np


class Neuron():

    def __init__(self, number_of_input, weight=None):
        # self.weight = np.random.rand(1, number_of_input)
        self.weight = weight if weight is not None else np.zeros(number_of_input, dtype=np.float64)

    def outputs(self, tensor):
        # what we want is >> weight * inputs
        z = self.weight @ tensor
        self.outputs = self.sigmoid(z)
        return self
        # return self.sigmoid(z)

    def derivative(self, ys):
        self.ys = ys
        return ys

    # This is an activation function (it normalize our result)
    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))
