from collections import defaultdict

from beras.core import Diffable, Tensor
from collections import deque

class GradientTape:

    def __init__(self):
        # Dictionary mapping the object id of an output Tensor to the Diffable layer it was produced from.
        self.previous_layers: defaultdict[int, Diffable | None] = defaultdict(lambda: None)

    def __enter__(self):
        # When tape scope is entered, all Diffables will point to this tape.
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # When tape scope is exited, all Diffables will no longer point to this tape.
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        """

        ### TODO: Populate the grads dictionary with {weight_id, weight_gradient} pairs and return the gradients of the sources

        # use a tensor/contribution pair to account for branch/dif length paths
        queue = deque([(target, None)])                  ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        # Use id(tensor) to get the object id of a tensor object.
        # in the end, your grads dictionary should have the following structure:
        # {id(tensor): [gradient]}

        # What tensor and what gradient is for you to implement!
        # compose_input_gradients and compose_weight_gradients are methods that will be helpful
        while queue:
            current, current_change = queue.popleft()
            layer = self.previous_layers[id(current)]
            if layer is None:
                continue
            inputs = layer.inputs
            input_composition = layer.compose_input_gradients(current_change)
            # only append for true inputs not weights
            for input, composition in zip(inputs, input_composition):
                # initialize if gradient is not there, otherwise append
                key = id(input)
                if grads[key] is None:
                    grads[key] = []
                grads[key].append(composition)
                queue.append((input, [composition]))
            if layer.weights:
                weights = layer.weights
                weight_compositions = layer.compose_weight_gradients(current_change)
                for weight, weight_composition in zip(weights, weight_compositions):
                    key = id(weight)
                    if grads[key] is None:
                        grads[key] = []
                    grads[key].append(weight_composition)
        res = []
        for source in sources:
            res.append(Tensor(sum(grads[id(source)])))
        return res


