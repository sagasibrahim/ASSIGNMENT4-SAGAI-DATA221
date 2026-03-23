from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print("Training Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

importances = model.feature_importances_
indices = np.argsort(importances)[-5:]

print("\nTop 5 Features:")
for i in indices[::-1]:
    print(data.feature_names[i], importances[i])

# Discussion
# By limiting the depth of the tree, I’m forcing the model to stay simpler,
# which helps prevent it from overfitting to the training data.
#
# Feature importance is useful because it shows which features the model is actually relying on.
# That makes decision trees easier to understand compared to a lot of other models.