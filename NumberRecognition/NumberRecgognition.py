import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

#Loading dataset, using the MINST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#Want the inputs to be between 0 and 1, normalizing
x_train = x_train / 255
x_test = x_test /255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam",
              loss = "sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(x_train, y_train)

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

plt.imshow(x_test[0], cmap="grey")
plt.title(f"True label : {y_test[0]}")
plt.show()

predictions = model.predict(x_test[:1])
print("predicted digit", predictions.argmax())
