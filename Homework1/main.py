import numpy as np
from sklearn.linear_model import LinearRegression
from TheoryGraph import plot_j_vs_x1


X = np.array([-2, -5, -3, 0, -8, -2, 1, 5, -1, 6]).reshape(-1, 1)
Y = np.array([1, -4, 1, 3, 11, 5, 0, -1, -3, 1]).reshape(-1, 1)  # 直接reshape

X_aug = np.hstack([np.ones((X.shape[0], 1)), X])

#print(X_aug)
#print(Y)

theta = np.linalg.inv(X_aug.T @ X_aug) @ (X_aug.T @ Y)
print("Theory: ")
print("Question 1 (a): ")
print("Theta:\n", theta, "\n")
print("Model Equation: y =", theta[0, 0], "+", theta[1, 0], "* x")


model = LinearRegression()
model.fit(X, Y)

print("Question 1  (b): ")
print("Theta_0:", model.intercept_[0])
print("Theta_1:", model.coef_[0])

print("==============================================")

print("Question 2 (a): ")
print("Both 2(x_1 + x_2 - 2)")

print("Question 2 (b): ")
plot_j_vs_x1()


print("Question 2 (c): ")

