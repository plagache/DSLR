import numpy as np

# here we should have only numpy array/object


# Brain input features_tensor, labels_tensor
# create weights_matrix with shape
class Brain():
    def __init__(self, xs, classes, features, ys=None, weights=None):
        self.xs = xs
        self.classes = classes
        self.features = features
        if weights is not None:
            self.weights = weights
        else:
            self.ys = ys
            self.weights =  np.zeros((len(self.classes), len(self.features)), dtype=np.float64)

    def test_shape(self):
        if (len(self.classes),len(self.features)) == self.weights.shape:
            print("shape is equal, classes:", len(self.classes), "features:", len(self.features), "weights:", self.weights.shape)

# class Neuron():
    # def __init__(self, number_of_input, weight=None):
    # def __init__(self, xs, name=None, weight=None):
    #     self.name = name
    #     self.xs = xs
    #     # self.number_of_input = xs.T.shape
    #     self.number_of_input = len(xs.T[0])
    #     self.weight = weight if weight is not None else np.zeros(self.number_of_input, dtype=np.float64)

    # def outputs(self, tensor):
        # self._z = self.weights @ tensor
    def predictions(self):
        self._z = self.weights @ self.xs.T
        self._outputs = self.logistic(self._z)
        return self._outputs

    # This is an activation function of the sigmoid type
    # in our case a logistic one (it normalize our result between 0 and 1)
    # https://en.wikipedia.org/wiki/Logistic_function
    def logistic(self, z):
        return 1.0 / (1 + np.exp(-z))

    def diffs(self):
        self._diffs = self._outputs - self.ys
        return self._diffs

    def loss(self):
        self.log_loss_sum = np.log(np.where(self.ys == 1.0, self.predictions(), 1 - self.predictions())).sum()
        # self.log_loss_sum = np.log(np.where(self.ys == 1.0, self._outputs, 1 - self._outputs)).sum()
        self._loss = - self.log_loss_sum / len(self.xs)
        return self._loss

    def update_weights(self, learning_rate):
        self._derivative = self.diffs() @ self.xs
        self._derivative /= len(self.xs)
        self.weights -= learning_rate * self._derivative
        return self.weights
