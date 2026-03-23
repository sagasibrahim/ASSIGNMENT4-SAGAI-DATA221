from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

print("Training Accuracy:", accuracy_score(y_train, train_pred))
print("Test Accuracy:", accuracy_score(y_test, test_pred))

# Discussion
# Entropy is basically a way of measuring how "mixed" or uncertain the data is.
# A good split is one that makes the groups more pure (less mixed).
#
# If the training accuracy ends up much higher than the test accuracy,
# that usually means the model is overfitting and memorizing the training data.
# If they’re close, then the model is generalizing pretty well.