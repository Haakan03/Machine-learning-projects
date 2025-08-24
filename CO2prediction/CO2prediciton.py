import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np



csv_path = "CO2prediction/CO2Emissions_Canada.csv"
df = pd.read_csv(csv_path)

print(df.head())


X = df[["Engine Size(L)", "Cylinders", "Fuel Consumption City (L/100 km)", "Fuel Consumption Hwy (L/100 km)", "Fuel Consumption Comb (L/100 km)"]]
Y = df["CO2 Emissions(g/km)"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1)  # <--- bare 1 output, ingen aktiveringsfunksjon!
])

model.compile(optimizer="adam",
              loss="mse",
              metrics=["mae"])

history = model.fit(X_train, Y_train,
          epochs=20,
          validation_data=(X_test, Y_test),
          batch_size=32)

loss, mae = model.evaluate(X_test, Y_test)
print(f"Mean absolute error on test data: {mae:.2f}")

preds = model.predict(X_test[:5])
print(f"predicitons: {preds.flatten()}")
print(f"True values: {Y_test.values[:5]}")

history_dict = history.history

plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(history_dict["loss"], label="Train Loss")
plt.plot(history_dict["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.legend()
plt.title("Loss curves")

# MAE
plt.subplot(1,2,2)
plt.plot(history_dict["mae"], label="Train MAE")
plt.plot(history_dict["val_mae"], label="Val MAE")
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error")
plt.legend()
plt.title("MAE curves")

plt.show()

y_pred = model.predict(X_test).flatten()

plt.figure(figsize=(6,6))
plt.scatter(Y_test, y_pred, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()],
         [Y_test.min(), Y_test.max()],
         "r--")  # perfekt linje
plt.xlabel("True CO2 emissions")
plt.ylabel("Predicted CO2 emissions")
plt.title("True vs. Predicted")
plt.show()