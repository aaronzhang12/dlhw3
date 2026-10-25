from collections import defaultdict
import numpy as np

class BasicOptimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    def apply_gradients(self, trainable_params, grads):
        for param, grad in zip(trainable_params, grads):
            param.assign(param - self.learning_rate*grad)


class RMSProp:
    def __init__(self, learning_rate, beta=0.9, epsilon=1e-6):
        self.learning_rate = learning_rate
        self.beta = beta
        self.epsilon = epsilon
        self.v = defaultdict(lambda: 0)

    def apply_gradients(self, trainable_params, grads):
        for param, grad in zip(trainable_params, grads):
            key = id(param)

            self.v[key] = self.beta * self.v[key] + (1-self.beta) * (grad**2)
            param.assign(param - self.learning_rate*grad/(np.sqrt(self.v[key]) + self.epsilon))


class Adam:
    def __init__(
        self, learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7, amsgrad=False
    ):


        self.learning_rate = learning_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = defaultdict(lambda: 0)         # First moment zero vector
        self.v = defaultdict(lambda: 0)         # Second moment zero vector.
        self.t = 0                              # Time counter

    def apply_gradients(self, trainable_params, grads):
        self.t += 1
        for param, grad in zip(trainable_params, grads):
            key = id(param)

            self.m[key] = self.beta_1 * self.m[key] + (1-self.beta_1) * grad
            self.v[key] = self.beta_2 * self.v[key] + (1-self.beta_2) * (grad ** 2)

            m = self.m[key]/(1-self.beta_1**self.t)
            v = self.v[key]/(1-self.beta_2**self.t)

            param.assign(param - self.learning_rate * m/(np.sqrt(v) + self.epsilon))





