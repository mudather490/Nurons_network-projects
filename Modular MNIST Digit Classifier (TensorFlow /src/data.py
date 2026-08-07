import numpy as np
from tensorflow import keras
from typing import Tuple

def load_and_preprocess_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Loads the MNIST dataset and normalizes pixel intensities from [0, 255] to [0.0, 1.0].
    
    Returns:
        ((X_train, y_train), (X_test, y_test))
    """
    # Load 60,000 training and 10,000 test 28x28 grayscale images
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Cast uint8 integers (0 to 255) to float32 and scale to [0.0, 1.0].
    # Neural networks optimize faster and more stably when input features are zero-centered or scaled.
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    return (x_train, y_train), (x_test, y_test)
