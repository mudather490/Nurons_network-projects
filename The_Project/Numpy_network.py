import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=2, hidden_size=3, output_size=1):
        # Step 1: Initialize Parameters
        self.w1 = np.random.randn(hidden_size, input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.w2 = np.random.randn(output_size, hidden_size)
        self.b2 = np.zeros((output_size, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return z > 0

    def mse_loss(self, y_pred, y_true):
        return np.mean(np.square(y_pred - y_true))

    def forward(self, x):
        self.z1 = np.dot(self.w1, x) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.w2, self.a1) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        m = X.shape[1]
        dz2 = self.a2 - y
        dw2 = (1/m) * np.dot(dz2, self.a1.T)
        db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)
        da1 = np.dot(self.w2.T, dz2)
        dz1 = da1 * (self.a1 * (1 - self.a1))
        dw1 = (1/m) * np.dot(dz1, X.T)
        db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)
        return dw1, db1, dw2, db2

    def train(self, X, Y, iterations, alpha):
        for i in range(iterations):
            self.forward(X)
            dw1, db1, dw2, db2 = self.backward(X, Y)
            self.w1 -= alpha * dw1
            self.b1 -= alpha * db1
            self.w2 -= alpha * dw2
            self.b2 -= alpha * db2
            if i % 1000 == 0:
                loss = self.mse_loss(self.a2, Y)
                print(f"Iteration {i}, Loss: {loss:.6f}")

    def predict(self, X):
        a2 = self.forward(X)
        return np.round(a2)

    def evaluate(self, X, Y):
        predictions = self.predict(X)
        # True Positives, False Positives, False Negatives
        tp = np.sum((predictions == 1) & (Y == 1))
        fp = np.sum((predictions == 1) & (Y == 0))
        fn = np.sum((predictions == 0) & (Y == 1))
        tn = np.sum((predictions == 0) & (Y == 0))

        accuracy = (tp + tn) / Y.size
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall
        }

#forward propagation
def sigmoid(z) :
  return 1/(1+ np.exp(-z))

def forword_prop(w1, b1, w2, b2, x):
  z1 = np.dot(w1, x)+ b1
  a1 = sigmoid(z1)
  z2 = np.dot(w2, a1)+ b2
  a2 = sigmoid(z2)

  return z1, a1, z2, a2

# backward propagation
def backword_prop(w1, b1, w2, b2, z1, a1, z2, a2, X, y):
  m = X.shape[1]

  # Output layer gradients
  dz2= a2-y
  dw2= (1/m)*np.dot(dz2, a1.T)
  db2= (1/m)*np.sum(dz2, axis=1, keepdims=True)
   

  #Hidden Layer gradients 
  da1 = np.dot(w2.T, dz2)
  dz1 = da1 * (a1 *(1 - a1))
  dw1 = (1/m)*np.dot(dz1, X.T)
  db1 = (1/m)*np.sum(dz1, axis=1, keepdims=True)

  return dw1, db1, dw2, db2

def update_parms(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
  w1 = w1- alpha * dw1
  b1 = b1- alpha * db1
  w2 = w2 - alpha * dw2
  b2 = b2 - alpha * db2

  return w1, b1, w2, b2

# XOR Data
X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])

# Using the class
model = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)
model.train(X, Y, iterations=10000, alpha=0.5)

# Final predictions using the class instance
predictions = model.predict(X)
metrics = model.evaluate(X, Y)

print("Inputs:")
print(X)
print("\nPredicted Outputs (rounded):")
print(predictions)
print("\nPerformance Metrics:")
for metric, value in metrics.items():
    print(f"{metric.capitalize()}: {value:.2f}")

from sklearn.datasets import load_iris

# 1. Load data
data = load_iris()
X = data.data.T  # Transpose to shape (4, 150) to match (features, samples)
y = data.target.reshape(1, -1)  # Shape (1, 150)

# 2. Re-initialize model for Iris (4 inputs, 3 hidden, 1 output for binary version)
# Note: Iris typically has 3 classes; for this simple NN, let's just train on 4 features.
model = NeuralNetwork(input_size=4, hidden_size=5, output_size=1)

# 3. Train
model.train(X, y, iterations=5000, alpha=0.01)
  
