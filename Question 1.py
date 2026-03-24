from sklearn.datasets import load_breast_cancer
import numpy as np

data = load_breast_cancer()

X = data.data
y = data.target

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

unique, counts = np.unique(y, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

# Discussion:
# Dataset is slightly imbalanced
# Class balance matters because models can bias toward majority class