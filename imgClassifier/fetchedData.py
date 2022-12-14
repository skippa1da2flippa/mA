from sklearn.datasets import fetch_openml

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
y = y.astype(int)
X = X / 255

print(X)
