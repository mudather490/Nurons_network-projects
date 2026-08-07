import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_mlp_model(input_shape: tuple = (28, 28), num_classes: int = 10) -> keras.Model:
    """
    Constructs a Multi-Layer Perceptron (MLP) for multi-class digit classification.
    """
    model = keras.Sequential([
        # Explicit input layer specifying spatial dimensions (28x28 matrix per image)
        keras.Input(shape=input_shape),
        
        # Flattens 2D matrices (28x28) into a 1D vector of length 784 (28*28)
        layers.Flatten(),
        
        # Dense hidden layer with 128 neurons using Rectified Linear Unit (ReLU) activation
        # ReLU introduces non-linearity: f(x) = max(0, x)
        layers.Dense(128, activation='relu'),
        
        # Dropout randomly deactivates 20% of neurons during each training step
        # This prevents co-adaptation of neurons and reduces overfitting
        layers.Dropout(0.2),
        
        # Output layer with 10 units representing class logits, converted to probabilities via Softmax
        # Softmax outputs a probability distribution where all class values sum to 1.0
        layers.Dense(num_classes, activation='softmax')
    ], name="digit_classifier")
    
    return model
