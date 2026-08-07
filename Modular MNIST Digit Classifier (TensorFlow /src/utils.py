import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

def plot_training_history(history: keras.callbacks.History) -> None:
    """Generates train vs validation loss and accuracy line graphs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves monitor underfitting and overfitting
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves monitor performance trends over training epochs
    ax2.plot(history.history['accuracy'], label='Train Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def evaluate_model_performance(model: keras.Model, x_test: np.ndarray, y_test: np.ndarray) -> None:
    """Computes Precision, Recall, F1-Score, and plots the confusion matrix."""
    # Predict continuous probabilities for each class
    y_pred_probs = model.predict(x_test, verbose=0)
    
    # Take argmax across columns to select class with highest predicted probability
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, digits=4))
    
    # Confusion matrix visualizes misclassifications (e.g., distinguishing digit 4 from 9)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('MNIST Test Confusion Matrix')
    plt.show()
