import numpy as np

class Perceptron:
    def __init__(self, n_features: int, learning_rate: float = 0.01):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.eta = learning_rate

    def prediction_for(self, input: np.ndarray) -> int:
        sum = np.dot(input, self.w) + self.b
        return 1 if sum >= 0 else 0

    def train_step (
        self, 
        input: np.ndarray, 
        target: int, 
        prediction: int
    ):
        err = target - prediction
        self.w += self.eta * err * input
        self.b += self.eta * err * 1
