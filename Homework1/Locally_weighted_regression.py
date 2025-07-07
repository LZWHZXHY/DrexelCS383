import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ReadFile import *

def locally_weighted_regression(X_train, y_train, X_test, y_test, k=1.0):
    N_test = X_test.shape[0]
    y_preds = []

    for i in range(N_test):
        x_query = X_test[i:i+1]  # shape (1, d)


        dists = np.sum(np.abs(X_train - x_query), axis=1)  # shape (N_train,)


        weights = np.exp(-dists / (k ** 2))  # shape (N_train,)


        W = np.diag(weights)


        X_aug = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        x_query_aug = np.hstack(([1.0], x_query.flatten())).reshape(1, -1)


        try:
            theta = np.linalg.inv(X_aug.T @ W @ X_aug) @ X_aug.T @ W @ y_train
        except np.linalg.LinAlgError:
            theta = np.linalg.pinv(X_aug.T @ W @ X_aug) @ X_aug.T @ W @ y_train  # 使用伪逆避免奇异矩阵


        y_pred = x_query_aug @ theta
        y_preds.append(y_pred.item())


    y_preds = np.array(y_preds).reshape(-1, 1)
    rmse = np.sqrt(np.mean((y_test - y_preds) ** 2))
    return rmse, y_preds

def run_part3_locally_weighted_lr():

    df = pd.read_csv("x06Simple.csv", skiprows=1, header=None)
    data = df.iloc[:, 1:].to_numpy()  # 去掉 index 列


    np.random.seed(0)
    np.random.shuffle(data)


    N = data.shape[0]
    train_size = int(np.ceil(2 * N / 3))
    train_data = data[:train_size]
    test_data = data[train_size:]

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1].reshape(-1, 1)
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1].reshape(-1, 1)


    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std


    rmse, y_preds = locally_weighted_regression(X_train_std, y_train, X_test_std, y_test)


    print("=== Part 3: Locally Weighted Linear Regression ===")
    print(f"Test RMSE: {rmse:.4f}")


