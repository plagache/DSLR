import numpy as np
from numpy.typing import NDArray

from nn import Brain
from variables import learning_rate_decay, scheduler

# differente class of hyperparamater optimization

# https://en.wikipedia.org/wiki/Hyperparameter_optimization


# should get the all Tensor same as GD, and select an item randomly
def stochastic_gradient_descent(brain: Brain, learning_rate: float, tensor: NDArray[np.float64], labels_tensor: NDArray[np.float64]):
    # we want to select an element of the tensor randomly and do the prediction
    random_index = np.random.randint(tensor.shape[0])  # Randomly select an index along the first axis
    selected_item = tensor[random_index]
    selected_label = labels_tensor.T[random_index]

    brain.predictions(selected_item)
    brain.diffs(selected_label)
    brain.loss(selected_item, selected_label)
    brain.update_weights(selected_item, learning_rate)

    return brain._loss


def gradient_descent(brain: Brain, learning_rate: float, tensor: NDArray[np.float64], labels_tensor: NDArray[np.float64]):
    brain.predictions(tensor)
    brain.diffs(labels_tensor)
    brain.loss(tensor, labels_tensor)
    brain.update_weights(tensor, learning_rate)
    return brain._loss


def learning_rate_scheduler(initial_learning_rate, step):
    match scheduler:
        case None:
            return initial_learning_rate
        case "linear":
            return linear_scheduler(initial_learning_rate, step)
        case "exp":
            return exponential_scheduler(initial_learning_rate, step)
        case _:
            raise ValueError(f"scheduler can only be 'exp' or 'linear'\nscheduler == '{scheduler}'")


def exponential_scheduler(initial_learning_rate, step):
    return initial_learning_rate * np.exp(-learning_rate_decay * step)

def linear_scheduler(initial_learning_rate, step):
    return (1 / (1 + learning_rate_decay * step)) * initial_learning_rate
