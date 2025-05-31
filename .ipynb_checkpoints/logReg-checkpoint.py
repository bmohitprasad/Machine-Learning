import numpy as np
import pandas as pd

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression training function
def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    X = np.array(X)
    y = np.array(y)

    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for i in range(epochs):
        linear_model = np.dot(X, weights) + bias
        y_pred = sigmoid(linear_model)

        # Gradient computation
        dw = np.dot(X.T, (y_pred - y)) / n_samples
        db = np.sum(y_pred - y) / n_samples

        # Parameter update
        weights -= lr * dw
        bias -= lr * db

    return weights, bias

# Prediction function
def predict(X, weights, bias):
    X = np.array(X)
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    return [1 if p >= 0.5 else 0 for p in y_pred]

# Load your CSV data (example: stock_data.csv)
df = pd.read_csv("dataset/tesla-stock-price.csv")

# Convert date to datetime (optional)
df['date'] = pd.to_datetime(df['date'], dayfirst=True)

# Sort by date to maintain time order (important for target labeling)
df = df.sort_values(by="date").reset_index(drop=True)

# Create binary target: 1 if next day's close > today’s close, else 0
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# Drop last row (no label possible for it)
df = df.dropna()

# Feature selection
features = df[['open', 'high', 'low', 'volume']]
target = df['target']

# Normalize features for better convergence
features = (features - features.mean()) / features.std()

# Train the model
weights, bias = train_logistic_regression(features, target, lr=0.01, epochs=1000)

# Prediction example
predictions = predict(features, weights, bias)

# Accuracy
accuracy = np.mean(predictions == target)
print(f"Training Accuracy: {accuracy * 100:.2f}%")