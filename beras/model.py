from abc import abstractmethod
from collections import defaultdict
from typing import Union

from beras.core import Diffable, Tensor, Callable
from beras.gradient_tape import GradientTape
import numpy as np
import math

def print_stats(stat_dict:dict, batch_num=None, num_batches=None, epoch=None, avg=False):
    """
    Given a dictionary of names statistics and batch/epoch info,
    print them in an appealing manner. If avg, display stat averages.

    :param stat_dict: dictionary of metrics to display
    :param batch_num: current batch number
    :param num_batches: total number of batches
    :param epoch: current epoch number
    :param avg: whether to display averages
    """
    title_str = " - "
    if epoch is not None:
        title_str += f"Epoch {epoch+1:2}: "
    if batch_num is not None:
        title_str += f"Batch {batch_num+1:3}"
        if num_batches is not None:
            title_str += f"/{num_batches}"
    if avg:
        title_str += f"Average Stats"
    print(f"\r{title_str} : ", end="")
    op = np.mean if avg else lambda x: x
    print({k: np.round(op(v), 4) for k, v in stat_dict.items()}, end="")
    print("   ", end="" if not avg else "\n")


def update_metric_dict(super_dict: dict, sub_dict: dict):
    """
    Appends the average of the sub_dict metrics to the super_dict's metric list

    :param super_dict: dictionary of metrics to append to
    :param sub_dict: dictionary of metrics to average and append
    """
    for k, v in sub_dict.items():
        super_dict[k] += [np.mean(v)]


class Model(Diffable):

    def __init__(self, layers: list[Diffable]):
        """
        Initialize all trainable parameters and take layers as inputs
        """
        # Initialize all trainable parameters
        self.layers = layers

    @property
    def weights(self) -> list[Tensor]:
        """
        Return the weights of the model by iterating through the layers
        """
        res = []
        for layer in self.layers:
            res.extend(layer.weights)
        return res


    def compile(self, optimizer: Diffable, loss_fn: Diffable, acc_fn: Callable):
        """
        "Compile" the model by taking in the optimizers, loss, and accuracy functions.
        In more optimized DL implementations, this will have more involved processes
        that make the components extremely efficient but very inflexible.
        """
        self.optimizer      = optimizer
        self.compiled_loss  = loss_fn
        self.compiled_acc   = acc_fn

    def fit(self, x: Tensor, y: Union[Tensor, np.ndarray], epochs: int, batch_size: int):
        """
        Trains the model by iterating over the input dataset and feeding input batches
        into the batch_step method with training. At the end, the metrics are returned.
        """
        history = defaultdict(list)
        for i in range(epochs):
            epoch_stats = defaultdict(list)
            for j in range(math.ceil(len(x) / batch_size)):
                batch_start = j*batch_size
                batch_end = min((j+1)*batch_size, len(x))
                batch_x = x[batch_start: batch_end, ]
                batch_y = y[batch_start: batch_end, ]
                metrics = self.batch_step(batch_x, batch_y, training = True)
                # get batch metrics and add to the current epoch stats
                update_metric_dict(epoch_stats, metrics)
                print_stats(metrics, batch_num=j, epoch = i, avg = False)
            print_stats(epoch_stats, epoch = i, avg = True)
            update_metric_dict(history, epoch_stats)
        return history

    def evaluate(self, x: Tensor, y: Union[Tensor, np.ndarray], batch_size: int):
        """
        X is the dataset inputs, Y is the dataset labels.
        Evaluates the model by iterating over the input dataset in batches and feeding input batches
        into the batch_step method. At the end, the metrics are returned. Should be called on
        the testing set to evaluate accuracy of the model using the metrics output from the fit method.

        NOTE: This method is almost identical to fit (think about how training and testing differ --
        the core logic should be the same)
        """
        evaluation_stats = defaultdict(list)
        prediction_batches = []
        batch_sizes = []
        for j in range(math.ceil(x.shape[0] / batch_size)):
            batch_start = j*batch_size
            batch_end = max((j+1)*batch_size, x.shape[0])
            batch_x = x[batch_start: batch_end, ]
            batch_y = y[batch_start: batch_end, ]
            batch_metrics, batch_predictions = self.batch_step(batch_x, batch_y, training = False)
            update_metric_dict(evaluation_stats, batch_metrics)
            prediction_batches.append(batch_predictions)
            batch_sizes.append(len(batch_x))

        # give the whole dataset metrics and predictions
        metrics = {
            name: np.average(values, weights=batch_sizes)
            for name, values in evaluation_stats.items()
        }
        predictions = np.concatenate(prediction_batches, axis = 0)
        print_stats(metrics, avg = True)
        return metrics, predictions

    def get_input_gradients(self) -> list[Tensor]:
        return super().get_input_gradients()

    def get_weight_gradients(self) -> list[Tensor]:
        return super().get_weight_gradients()
    
    @abstractmethod
    def batch_step(self, x: Tensor, y: Tensor, training:bool =True) -> dict[str, float]:
        """
        Computes loss and accuracy for a batch. This step consists of both a forward and backward pass.
        If training=false, don't apply gradients to update the model! Most of this method (, loss, applying gradients)
        will take place within the scope of Beras.GradientTape()
        """
        raise NotImplementedError("batch_step method must be implemented in child class")

class SequentialModel(Model):
    def forward(self, inputs: Tensor) -> Tensor:
        """Forward pass in sequential model. It's helpful to note that layers are initialized in beras.Model, and
        you can refer to them with self.layers. You can call a layer by doing var = layer(input).
        """
        output = inputs
        for layer in self.layers:
            output = layer(output)
        return output

    def batch_step(self, x:Tensor, y: Tensor, training: bool =True) -> dict[str, float]:
        """Computes loss and accuracy for a batch. This step consists of both a forward and backward pass.
        If training=false, don't apply gradients to update the model! Most of this method (, loss, applying gradients)
        will take place within the scope of Beras.GradientTape()"""
        ## TODO: Compute loss and accuracy for a batch. Return as a dictionary
        ## If training, then also update the gradients according to the optimizer
        with GradientTape() as tape:
            predictions = self.forward(x)
            loss = self.compiled_loss(predictions, y)
        accuracy = self.compiled_acc(predictions, y)

        if training:
            parameters = self.trainable_variables
            gradients = tape.gradient(loss, parameters)
            self.optimizer.apply_gradients(parameters, gradients)

        metrics = {"loss": loss, "acc": accuracy}
        if training:
            return metrics
        else:
            return metrics, predictions
