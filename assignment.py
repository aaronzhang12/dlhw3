from pathlib import Path

import numpy as np

from beras.activations import ReLU, Softmax
from beras.layers import Dense
from beras.losses import CategoricalCrossEntropy
from beras.metrics import CategoricalAccuracy
from beras.optimizers import Adam
from preprocess import load_and_preprocess_data

from beras.model import SequentialModel

def get_model():
    model = SequentialModel(
        [
            Dense(784, 128, initializer="kaiming"),
            ReLU(),
            Dense(128, 64, initializer="kaiming"),
            ReLU(),
            Dense(64, 10, initializer="xavier"),
            Softmax(),
        ]
    )
    return model

def get_optimizer():
    return Adam(learning_rate=1e-3)

def get_loss_fn():
    return CategoricalCrossEntropy()

def get_acc_fn():
    return CategoricalAccuracy()

if __name__ == "__main__":
    epochs = 8
    batch_size = 64

    model = get_model()
    model.compile(get_optimizer(), get_loss_fn(), get_acc_fn())
    train_x, train_y, test_x, test_y = load_and_preprocess_data()

    history = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size)
    print("\nTest results:")
    metrics, predictions = model.evaluate(test_x, test_y, batch_size=batch_size)
    np.save(Path(__file__).resolve().parent / "predictions.npy", predictions)
