# differente class of hyperparamater optimization

# https://en.wikipedia.org/wiki/Hyperparameter_optimization


from nn import Neuron


def Optimizer():
    def __init__(self, neural_network: Neuron, learning_rate: float):
        self.nn = neural_network
        self.params = neural_network.weight
        self.lr = learning_rate


# def GradientDescent():
#
#
# def SGD():
#
#
