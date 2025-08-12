import kagglehub
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class LinearRegression:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        #Using minste kvadraters metode (husker ikke engelske navnet)
        X = np.c_[np.ones(X.shape[0]), X]
        self.weights = np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X) @ y

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.weights

def pred_house_prices(x_features, y_feature):
    path = kagglehub.dataset_download("yasserh/housing-prices-dataset")

    csv_path = os.path.join(path, "Housing.csv")

    df = pd.read_csv(csv_path)
   
    X = df[x_features].values
    y = df[y_feature].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    loss = sum([int((y_pred[i]-y_test[i])**2) for i in range(len(y_test))])
    plt.figure()
    plt.title(f"Features: {x_features}, Loss = {loss}")
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual price")
    plt.ylabel("Predicted price")

    # Legger til linjen f(x) = x
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')  # rød stiplet linje

    # Setter like aksegrenser
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)

    plt.gca().set_aspect('equal', adjustable='box')  # Sørger for at dimensjonene er like

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
