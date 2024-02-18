import numpy as np

from nn import Brain

# differente class of hyperparamater optimization

# https://en.wikipedia.org/wiki/Hyperparameter_optimization


# should get the all Tensor same as GD, and select an item randomly
def sgd(tensor, neuron, ys, learning_rate):
    # input: tensor, ys, neuron.weight
    # output: outputs, diffs, losses
    outputs = neuron.outputs(tensor.T)
    diffs = outputs - ys

    loss = -np.log(np.where(ys == 1.0, outputs, 1 - outputs))

    # Update weight
    # inputs: neuron.weight, diffs, tensor
    # outputs: neuron.weight.updater
    # diffs @ tensor = [13, 1]
    diffnp = np.array([diffs]).reshape((1, 1))
    tensornp = np.reshape(tensor, (1, 13))
    derivative = diffnp @ tensornp
    derivative = np.reshape(derivative, (13,))
    neuron.weight -= learning_rate * derivative
    return loss


def gd(brain: Brain, learning_rate, tensor, labels_tensor):
    brain.predictions(tensor)
    brain.diffs(labels_tensor)
    brain.loss(tensor, labels_tensor)
    brain.update_weights(tensor, learning_rate)
    return brain._loss
