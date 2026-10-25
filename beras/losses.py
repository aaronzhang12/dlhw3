import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return Tensor(np.mean((y_pred - y_true) ** 2))

    def get_input_gradients(self) -> list[Tensor]:
        y_pred, y_true = self.inputs
        return [Tensor(2*(y_pred-y_true)/y_pred.size), Tensor(np.zeros_like(y_true))]

class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true) -> Tensor:
        """Categorical cross entropy forward pass!"""
        eps = 1e-7
        p = np.clip(y_pred, a_min = eps, a_max = 1-eps)
        per_sample_losses = -np.sum(y_true * np.log(p), axis = -1)
        return Tensor(np.mean(per_sample_losses))

    def get_input_gradients(self) -> list[Tensor]:
        """Categorical cross entropy input gradient method!"""
        eps = 1e-7
        y_pred, y_true = self.inputs
        p = np.clip(y_pred, eps, 1-eps)
        prediction_gradient = -y_true/(y_pred.shape[0] * p)
        return [Tensor(prediction_gradient), Tensor(np.zeros_like(y_true))]

