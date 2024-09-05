import numpy as np

# here we should have only numpy array/object


# Brain input: the class to detect, the features to train
# create parameters matrix with shape
class Brain:
    def __init__(self, classes, features, weights=None):
        self.classes = classes
        self.class_number: int = len(self.classes)
        self.features = features
        self.feature_number: int = len(self.features)
        self.weights = weights if weights is not None else np.zeros((self.class_number, self.feature_number), dtype=np.float64)

    # This is an activation function of the sigmoid type
    # in our case a logistic one (it normalize our result between 0 and 1)
    # https://en.wikipedia.org/wiki/Logistic_function
    def logistic(self, z):
        return 1.0 / (1 + np.exp(-z))

    def predictions(self, xs):
        self._z = self.weights @ xs.T
        self._outputs = self.logistic(self._z)
        return self._outputs

    def diffs(self, ys):
        self._diffs = self._outputs - ys
        return self._diffs

    def loss(self, xs, ys):
        self.log_loss = np.log(np.where(ys == 1.0, self._outputs, 1 - self._outputs))
        if len(xs.shape) == 1:
            self._loss = -self.log_loss
        else:
            self._loss = -self.log_loss.sum(axis=1) / len(xs)
        return self._loss

    def update_weights(self, xs, learning_rate):
        if len(xs.shape) == 1:
            self._diffs = self._diffs.reshape(-1, 1)  # Reshape to (4, 1)
            xs = xs.reshape(1, -1)
        self._derivative = self._diffs @ xs
        self._derivative /= len(xs)
        self.weights -= learning_rate * self._derivative
        return self.weights

    # refacto 
    # def __repr__(self):
    #     return f"{self.classes}\nNumber of Neurons: {self.class_number}"
    def neurons(self):
        print(self.classes)
        print("Number of Neurons: ", self.class_number)
        if (self.class_number, self.feature_number) == self.weights.shape:
            print(
                "Shape is Good| classes:",
                self.class_number,
                "| features:",
                self.feature_number,
                "| weights:",
                self.weights.shape,
            )
        for index, class_name in enumerate(self.classes):
            print(class_name, ":")
            print(self.weights[index])
