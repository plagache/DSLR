import numpy as np
# differente class of hyperparamater optimization

# https://en.wikipedia.org/wiki/Hyperparameter_optimization


from nn import Neuron


def Optimizer():
    def __init__(self, neural_network: Neuron, learning_rate: float):
        self.nn = neural_network
        self.params = neural_network.weight
        self.lr = learning_rate


def sgd(tensor, neuron, ys, learning_rate):
    # Separate tensor and ys here
    # input: tensor, ys, neuron.weight
    # output: outputs, diffs, losses
    row_count = len(tensor)
    loss = 0

    outputs = neuron.outputs(tensor.T)
    diffs = outputs - ys

    # for element, y in zip(outputs, ys):
    #     loss += y * np.log(element) + (1 - y) * np.log(1 - element)
    # loss = - loss / row_count
    loss = ys * np.log(outputs) + (1 - ys) * np.log(1 - outputs)
    loss = - loss / row_count

    # Update weight
    # inputs: neuron.weight, diffs, tensor
    # outputs: neuron.weight.updater
    # diffs @ tensor = [13, 1]
    diffnp = np.array([diffs]).reshape((1, 1))
    tensornp = np.reshape(tensor, (1, 13))
    derivative = diffnp @ tensornp
    derivative /= row_count
    derivative = np.reshape(derivative, (13,))
    neuron.weight -= learning_rate * derivative
    return loss


def gd(tensor, neuron, ys, learning_rate):
    # Separate tensor and ys here
    # input: tensor, ys, neuron.weight
    # output: outputs, diffs, losses
    row_count = len(tensor)
    loss = 0

    outputs = neuron.outputs(tensor.T)
    diffs = outputs - ys

    for element, y in zip(outputs, ys):
        loss += y * np.log(element) + (1 - y) * np.log(1 - element)
    loss = - loss / row_count

    # Update weight
    # inputs: neuron.weight, diffs, tensor
    # outputs: neuron.weight.updater
    # diffs @ tensor = [13, 1]
    derivative = diffs @ tensor
    derivative /= row_count
    neuron.weight -= learning_rate * derivative
    return loss
