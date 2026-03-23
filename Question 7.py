import ssl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model

ssl._create_default_https_context = ssl._create_unverified_context
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

X_test = X_test / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)
try:
    model = load_model("cnn_model.h5")
except:
    print("Error: cnn_model.h5 not found. Make sure to save the model in Question 6.")

predictions = np.argmax(model.predict(X_test), axis=1)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
misclassified = np.where(predictions != y_test)[0]

plt.figure(figsize=(10, 4))
for i, index in enumerate(misclassified[:3]):
    plt.subplot(1, 3, i + 1)
    plt.imshow(X_test[index].reshape(28, 28), cmap='gray')
    plt.title(f"True: {y_test[index]}, Pred: {predictions[index]}")
    plt.axis('off')
plt.show()