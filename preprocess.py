import numpy as np
from beras.onehot import OneHotEncoder
from beras.core import Tensor
from tensorflow.keras import datasets

def load_and_preprocess_data() -> tuple[Tensor, Tensor, Tensor, Tensor]:
    '''This is where we load in and preprocess our data! We load in the data 
        for you but you'll need to flatten the images, normalize the values and 
        convert the input images from numpy arrays into tensors
    Return the preprocessed training and testing data and labels!'''
    
    #Load in the training and testing data from the MNIST dataset
    (train_inputs, train_labels), (test_inputs, test_labels) = datasets.mnist.load_data()

    ## TODO: Flatten (reshape) and normalize the inputs
    ## Hint: train and test inputs are numpy arrays so you can use np methods on them!
    
    train_inputs = Tensor(train_inputs.reshape(train_inputs.shape[0], -1).astype(np.float32)/255.0)
    test_inputs = Tensor(test_inputs.reshape(test_inputs.shape[0], -1).astype(np.float32)/255.0)

    encoder = OneHotEncoder()
    encoder.fit(train_labels)

    train_labels = encoder(train_labels)
    test_labels = encoder(test_labels)

    return train_inputs, train_labels, test_inputs, test_labels


    ## TODO: Convert all of the data into Tensors (constructor in core.py)

