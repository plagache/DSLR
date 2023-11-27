import numpy as np



class Neuron():

    def __init__(self, number_of_input, weight=None):
        # self.weight = np.random.rand(1, number_of_input)
        self.weight = weight if weight is not None else np.zeros(number_of_input, dtype=np.float64)

    def outputs(self, tensor):
        # what we want is >> weight * inputs
        z = self.weight @ tensor
        return self.sigmoid(z)

    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))
