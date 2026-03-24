from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)
dt_cm = confusion_matrix(y_test, dt.predict(X_test))

plt.figure()
sns.heatmap(dt_cm, annot=True, fmt='d')
plt.title("Decision Tree CM")
plt.show()

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

nn = MLPClassifier(max_iter=500)
nn.fit(X_train_s, y_train)
nn_cm = confusion_matrix(y_test, nn.predict(X_test_s))

plt.figure()
sns.heatmap(nn_cm, annot=True, fmt='d')
plt.title("Neural Network CM")
plt.show()

# Discussion:
# DT = interpretable but can overfit
# NN = powerful but less interpretable