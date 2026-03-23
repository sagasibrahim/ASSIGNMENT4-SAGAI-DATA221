from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=42)
nn.fit(X_train_scaled, y_train)
nn_pred = nn.predict(X_test_scaled)

print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, tree_pred))
print("\nNeural Network Confusion Matrix:\n", confusion_matrix(y_test, nn_pred))

# Discussion
# If I had to choose, I’d probably go with the neural network for better performance,
# but the decision tree is easier to understand.
#
# Decision trees are nice because you can actually see how decisions are made,
# but they can overfit pretty easily.
#
# Neural networks are more powerful and flexible,
# but they’re basically a black box and harder to interpret.