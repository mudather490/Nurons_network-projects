import os
from src.data import load_and_preprocess_data
from src.model import build_mlp_model
from src.utils import plot_training_history, evaluate_model_performance

def main():
    # 1. Pipeline preparation
    (X_train, y_train), (X_test, y_test) = load_and_preprocess_data()
    
    # 2. Build & Compile
    model = build_mlp_model()
    model.compile(
        optimizer='adam',                       # Adaptive Moment Estimation optimizer
        loss='sparse_categorical_crossentropy', # Used when labels are integers, not one-hot encoded
        metrics=['accuracy']
    )
    model.summary()
    
    # 3. Train model with 10% held-out validation split
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )
    
    # 4. Diagnostics & Evaluation
    plot_training_history(history)
    evaluate_model_performance(model, X_test, y_test)
    
    # 5. Native Keras v3 Model Persistence
    os.makedirs("models", exist_ok=True)
    model.save("models/mnist_classifier.keras")
    print("\nModel saved successfully to models/mnist_classifier.keras")

if __name__ == "__main__":
    main()
