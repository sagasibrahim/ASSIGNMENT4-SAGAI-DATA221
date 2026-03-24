from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_test = X_test / 255.0
X_test = X_test.reshape(-1, 28, 28, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
(X_train, y_train), _ = fashion_mnist.load_data()
X_train = X_train / 255.0
X_train = X_train.reshape(-1, 28, 28, 1)

model.fit(X_train, y_train, epochs=3)
preds = model.predict(X_test)
pred_classes = np.argmax(preds, axis=1)

cm = confusion_matrix(y_test, pred_classes)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("CNN Confusion Matrix")
plt.show()

misclassified = np.where(pred_classes != y_test)[0]

for i in range(3):
    idx = misclassified[i]
    plt.imshow(X_test[idx].reshape(28,28), cmap='gray')
    plt.title(f"True: {y_test[idx]} | Pred: {pred_classes[idx]}")
    plt.axis('off')
    plt.show()

# Discussion:
# Errors occur between similar clothing types
# Improve with deeper network or more training