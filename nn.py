import numpy as np

# here we should have only numpy array/object


# Brain input features_tensor, labels_tensor
# create weights_matrix with shape
class Brain():
    # def __init__(self, classes, features, ys=None, weights=None):
    def __init__(self, classes, features, weights=None):
        self.classes = classes
        self.class_number = len(self.classes)
        self.features = features
        self.feature_number = len(self.features)
        self.weights = weights if weights is not None else np.zeros((self.class_number, self.feature_number), dtype=np.float64)


    def neurons(self):
        print(self.classes)
        print("Number of Neurons: ", self.class_number)
        if (self.class_number,self.feature_number) == self.weights.shape:
            print("Shape is Good| classes:", self.class_number, "| features:", self.feature_number, "| weights:", self.weights.shape)
        for index, class_name in enumerate(self.classes):
            print(class_name, ":")
            print(self.weights[index])



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
    def predictions(self, xs):
        self._z = self.weights @ xs.T
        self._outputs = self.logistic(self._z)
        return self._outputs

    # This is an activation function of the sigmoid type
    # in our case a logistic one (it normalize our result between 0 and 1)
    # https://en.wikipedia.org/wiki/Logistic_function
    def logistic(self, z):
        return 1.0 / (1 + np.exp(-z))

    def diffs(self, ys):
        self._diffs = self._outputs - ys
        return self._diffs

    def loss(self, xs, ys):
        self.log_loss_sum = np.log(np.where(ys == 1.0, self._outputs, 1 - self._outputs)).sum(axis=1)
        self._loss = - self.log_loss_sum / len(xs)
        return self._loss

    def update_weights(self, xs, learning_rate):
        self._derivative = self._diffs @ xs
        self._derivative /= len(xs)
        self.weights -= learning_rate * self._derivative
        return self.weights
