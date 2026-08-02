# Custom Neural Network from Scratch in NumPy

A lightweight implementation of a 2-layer Neural Network built completely from scratch using **NumPy**. This project demonstrates core Deep Learning mechanics—including forward propagation, backpropagation, gradient descent, and custom evaluation metrics—without relying on heavy frameworks like PyTorch or TensorFlow.

---

## 📌 Features

* **Zero Deep Learning Frameworks**: Pure mathematical implementation using linear algebra in NumPy.
* **Core Mechanisms**:
  * Vectorized Forward Propagation
  * Manual Backpropagation & Derivative Calculation
  * Mean Squared Error (MSE) Loss
  * Training Loop with Gradient Descent
* **Performance Evaluation**: Custom implementations for Accuracy, Precision, and Recall.

---

## 🛠️ Mathematical Implementation

1. **Forward Propagation**:
   $$Z_1 = W_1 X + b_1 \quad \rightarrow \quad A_1 = \sigma(Z_1)$$
   $$Z_2 = W_2 A_1 + b_2 \quad \rightarrow \quad A_2 = \sigma(Z_2)$$

2. **Activation Function**:
   $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

3. **Loss Function (MSE)**:
   $$\mathcal{L} = \frac{1}{n} \sum (Y_{\text{pred}} - Y_{\text{true}})^2$$

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone [https://github.com/YOUR-USERNAME/numpy-neural-network.git](https://github.com/YOUR-USERNAME/numpy-neural-network.git)
cd numpy-neural-network
pip install -r requirements.txt
```

### 2. Training on the XOR Problem

```python
import numpy as np
from neural_network import NeuralNetwork

# Define XOR Dataset
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])

# Initialize model (2 Inputs, 3 Hidden Neurons, 1 Output)
model = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)

# Train model
model.train(X, Y, iterations=10000, alpha=0.5)

# Evaluate model
metrics = model.evaluate(X, Y)
print("Performance:", metrics)
```

---

## 📊 Notebook Demo

For interactive visualizations, training loss plots, and experiments (e.g., Iris Dataset), check out the [`demo.ipynb`](./demo.ipynb) file.
