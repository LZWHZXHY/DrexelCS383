import numpy as np
import pandas as pd
import matplotlib as mpl


mpl.use('TkAgg')
import matplotlib.pyplot as plt


def gradient_descent_regression(X_train, y_train, X_test, y_test, eta=0.01, max_iter=1000, tol=2 ** -23):

    X_train_aug = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test_aug = np.hstack((np.ones((X_test.shape[0], 1)), X_test))


    theta = np.random.uniform(-1, 1, (X_train_aug.shape[1], 1))


    train_rmse_history = []
    test_rmse_history = []

    prev_rmse = None
    iteration = 0


    while iteration < max_iter:

        y_train_pred = X_train_aug @ theta
        train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
        train_rmse_history.append(train_rmse)


        y_test_pred = X_test_aug @ theta
        test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
        test_rmse_history.append(test_rmse)


        if prev_rmse is not None:
            percent_change = abs((prev_rmse - train_rmse) / prev_rmse)
            if percent_change < tol:
                break

        prev_rmse = train_rmse


        gradients = (1 / len(X_train_aug)) * X_train_aug.T @ (X_train_aug @ theta - y_train)


        theta -= eta * gradients

        iteration += 1

    return theta, train_rmse_history, test_rmse_history, iteration


def run_gradient_descent_regression():

    #plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False


    df = pd.read_csv("x06Simple.csv", skiprows=1, header=None)
    data = df.iloc[:, 1:].to_numpy()


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


    theta, train_rmse_history, test_rmse_history, iterations = gradient_descent_regression(
        X_train_std, y_train, X_test_std, y_test
    )
    X_test_aug = np.hstack((np.ones((X_test_std.shape[0], 1)), X_test_std))
    y_pred = X_test_aug @ theta
    final_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))


    print("=== Gradient Descent Result ===")
    print(f"convergence at {iterations} times iterations")
    print(f"Final test RMSE: {final_rmse:.4f}")


    intercept = f"{theta[0, 0]:.4f}"
    coefficients = []
    for i in range(1, theta.shape[0]):
        coef = theta[i, 0]
        sign = "+" if coef >= 0 else ""
        term = f"{sign}{coef:.4f} * x{i}"
        coefficients.append(term)

    equation = "y = " + intercept + " " + " ".join(coefficients)

    print("\nFinal Model:")
    print(equation)


    plt.figure(figsize=(10, 6))
    plt.plot(train_rmse_history, label='Train RMSE', linewidth=2)
    plt.plot(test_rmse_history, label='Test RMSE', linewidth=2)
    plt.xlabel('Iteration times')
    plt.ylabel('RMSE')
    plt.title('RMSE changes')
    plt.legend()
    plt.grid(True)


    output_file = 'RMSE Curve.png'
    plt.savefig(output_file)
    plt.tight_layout()
    plt.show()

    print(f"\nRMSE Curve save as: {output_file}")



#if __name__ == "__main__":
#    run_gradient_descent_regression()