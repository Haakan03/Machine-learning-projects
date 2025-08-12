import kagglehub
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class GradientDescent:
    def __init__(self, lr=0.01, epochs=5000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.epochs):
            y_pred = X @ self.weights
            gradient = -2/n_samples * X.T @ (y - y_pred)
            self.weights -= self.lr * gradient

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.weights


def pred_house_prices(x_features, y_feature):
    path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
    csv_path = os.path.join(path, "Housing.csv")
    df = pd.read_csv(csv_path)

    X = df[x_features].values
    y = df[y_feature].values

    # Scale features for gradient descent
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = GradientDescent(lr=0.01, epochs=5000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")

    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

X = ["area", "bedrooms", "bathrooms", "parking"]
y = "price"
pred_house_prices(X, y)


X = ["area", "bedrooms", "bathrooms"]
y = "price"
pred_house_prices(X, y)

X = ["area"]
y = "price"
pred_house_prices(X, y)
