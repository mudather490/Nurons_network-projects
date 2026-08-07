# 🧠 Modular MNIST Digit Classifier (TensorFlow / Keras)

A modular, production-ready Multi-Layer Perceptron (MLP) pipeline designed to classify 28x28 grayscale handwritten digits using TensorFlow 2.x and Keras 3.

---

## 📌 Architectural Overview

Unlike typical notebook implementations, this project separates data ingestion, model declaration, diagnostic visualization, and training orchestration into discrete Python modules to ensure maintainability and testability.

```text
mnist-digit-classifier/
├── src/
│   ├── __init__.py
│   ├── data.py          # Data ingestion & scaling
│   ├── model.py         # Sequential MLP architecture with Dropout
│   └── utils.py         # Training curves & Confusion Matrix visualizers
├── models/              # Native Keras model persistence (.keras)
├── train.py             # Main entry point to orchestrate training
├── requirements.txt     # Environment dependencies
└── README.md            # Project documentation
