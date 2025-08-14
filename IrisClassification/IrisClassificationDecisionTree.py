import kagglehub
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


path = kagglehub.dataset_download("arshid/iris-flower-dataset")

csv_path = os.path.join(path, "Iris.csv")

df = pd.read_csv(csv_path)

X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
