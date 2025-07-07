import numpy as np
import pandas as pd


#Read File
df = pd.read_csv("x06Simple.csv", skiprows=1, header=None)
data = df.iloc[:, 1:].to_numpy()

#Randomizes the data
np.random.seed(0)
np.random.shuffle(data)

#Selets the Training data and Testing data
N = data.shape[0]
train_size = int(np.ceil(2 * N / 3))
train_data = data[:train_size]
test_data = data[train_size:]

X_train = train_data[:, :-1]
y_train = train_data[:, -1].reshape(-1, 1)

X_test = test_data[:, :-1]
y_test = test_data[:, -1].reshape(-1, 1)

#Standardizes the data
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

X_train_std = (X_train - mean) / std
X_test_std = (X_test - mean) / std

#Closed-form linear regression
X_train_aug = np.hstack((np.ones((X_train_std.shape[0], 1)), X_train_std))  # shape (30, 3)
theta = np.linalg.inv(X_train_aug.T @ X_train_aug) @ X_train_aug.T @ y_train

#Applies the solution to the testing samples
X_test_aug = np.hstack((np.ones((X_test_std.shape[0], 1)), X_test_std))
y_pred = X_test_aug @ theta

#computes the RMSE
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))


THETA = theta


def print_result(THETA, rmse, feature_names=None):
    print("=== Part 2: Closed-form Linear Regression ===")

    terms = [f"{theta[0, 0]:.4f}"]  # bias 项

    for i in range(1, theta.shape[0]):
        name = f"x{i}" if feature_names is None else feature_names[i - 1]
        terms.append(f"{theta[i, 0]:+.4f} * {name}")

    eq = "y = " + " ".join(terms)
    print("expression:")
    print(eq)
    print(f"\n RMSE: {rmse:.4f}")


