import numpy as np
# differente class of hyperparamater optimization

# https://en.wikipedia.org/wiki/Hyperparameter_optimization


def sgd(tensor, neuron, ys, learning_rate):
    # input: tensor, ys, neuron.weight
    # output: outputs, diffs, losses
    outputs = neuron.outputs(tensor.T)
    diffs = outputs - ys

    loss = - np.log(np.where(ys == 1.0, outputs, 1 - outputs))

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


def gd(tensor, neuron, ys, learning_rate):
    # input: tensor, ys, neuron.weight
    # output: outputs, diffs, losses
    row_count = len(tensor)

    outputs = neuron.outputs(tensor.T)
    diffs = outputs - ys

    log_loss_sum = np.log(np.where(ys == 1.0, outputs, 1 - outputs)).sum()
    loss = - log_loss_sum / row_count

    # Update weight
    # inputs: neuron.weight, diffs, tensor
    # outputs: neuron.weight.updater
    # diffs @ tensor = [13, 1]
    derivative = diffs @ tensor
    derivative /= row_count
    neuron.weight -= learning_rate * derivative
    return loss
