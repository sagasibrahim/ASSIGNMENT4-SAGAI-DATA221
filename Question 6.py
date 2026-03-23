import tensorflow as tf
from tensorflow.keras import layers, models

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# 2. Normalize pixel values to the range [0, 1]
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# 3. Reshape images to include the channel dimension (28x28x1)
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'), # Added one dense layer for better feature processing
    layers.Dense(10, activation='softmax')
])

# 5. Compile and Train
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training for 15 epochs
model.fit(X_train, y_train, epochs=15, batch_size=64)

# 6. Report the test accuracy
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")

# DISCUSSION
# - CNNs are preferred over fully connected networks for images because they
#   preserve spatial relationships. A flat network treats pixels as independent,
#   but a CNN uses filters to see how pixels relate to their neighbors.
# - The convolution layer is learning to detect "features" like the sharp
#   edges of a sleeve, the curve of a neckline, or the texture of a shoe sole.