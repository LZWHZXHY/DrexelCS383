import numpy as np

X = np.array([
    [-2],
    [-5],
    [-3],
    [0],
    [-8],
    [-2],
    [1],
    [5],
    [-1],
    [6]
])

Y = np.array([[1, -4,1,3,11,5,0,-1,-3,1]]).reshape(-1, 1)

X_aug = np.hstack([np.ones((X.shape[0], 1)), X])

print(X_aug)
print(Y)

theta = np.linalg.inv(X_aug.T @ X_aug) @ X_aug.T @ Y

print(theta)