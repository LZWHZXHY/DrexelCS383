import numpy as np
import pandas as pd
import matplotlib as mpl

# 设置后端后再导入pyplot
mpl.use('TkAgg')  # 使用TkAgg后端确保图片显示
import matplotlib.pyplot as plt


def gradient_descent_regression(X_train, y_train, X_test, y_test, eta=0.01, max_iter=1000, tol=2 ** -23):
    # 添加偏置项（常数项1）
    X_train_aug = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_test_aug = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

    # 参数初始化 [-1, 1] 区间
    theta = np.random.uniform(-1, 1, (X_train_aug.shape[1], 1))

    # 记录训练和测试RMSE历史
    train_rmse_history = []
    test_rmse_history = []

    prev_rmse = None  # 用于存储前一次迭代的RMSE
    iteration = 0

    # 梯度下降主循环
    while iteration < max_iter:
        # 计算训练集预测值和RMSE
        y_train_pred = X_train_aug @ theta
        train_rmse = np.sqrt(np.mean((y_train - y_train_pred) ** 2))
        train_rmse_history.append(train_rmse)

        # 计算测试集RMSE
        y_test_pred = X_test_aug @ theta
        test_rmse = np.sqrt(np.mean((y_test - y_test_pred) ** 2))
        test_rmse_history.append(test_rmse)

        # 收敛条件检查（第一次迭代后开始检查）
        if prev_rmse is not None:
            percent_change = abs((prev_rmse - train_rmse) / prev_rmse)
            if percent_change < tol:
                break

        prev_rmse = train_rmse  # 更新前次RMSE

        # 计算梯度（批量梯度下降）
        gradients = (1 / len(X_train_aug)) * X_train_aug.T @ (X_train_aug @ theta - y_train)

        # 参数更新
        theta -= eta * gradients

        iteration += 1

    return theta, train_rmse_history, test_rmse_history, iteration


def run_gradient_descent_regression():
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

    # 数据加载和预处理
    df = pd.read_csv("x06Simple.csv", skiprows=1, header=None)
    data = df.iloc[:, 1:].to_numpy()  # 移除索引列

    # 随机打乱数据（随机种子=0）
    np.random.seed(0)
    np.random.shuffle(data)

    # 划分训练集（2/3）和测试集（1/3）
    N = data.shape[0]
    train_size = int(np.ceil(2 * N / 3))
    train_data = data[:train_size]
    test_data = data[train_size:]

    # 分离特征和标签
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1].reshape(-1, 1)
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1].reshape(-1, 1)

    # 数据标准化（基于训练集统计）
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    X_train_std = (X_train - mean) / std
    X_test_std = (X_test - mean) / std

    # 执行梯度下降回归
    theta, train_rmse_history, test_rmse_history, iterations = gradient_descent_regression(
        X_train_std, y_train, X_test_std, y_test
    )

    # 计算最终测试集RMSE
    X_test_aug = np.hstack((np.ones((X_test_std.shape[0], 1)), X_test_std))
    y_pred = X_test_aug @ theta
    final_rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    # 打印结果
    print("=== 梯度下降线性回归结果 ===")
    print(f"收敛于 {iterations} 次迭代后")
    print(f"最终测试RMSE: {final_rmse:.4f}")

    # 构建模型方程（修正格式，避免多余正号）
    intercept = f"{theta[0, 0]:.4f}"
    coefficients = []
    for i in range(1, theta.shape[0]):
        coef = theta[i, 0]
        sign = "+" if coef >= 0 else ""
        term = f"{sign}{coef:.4f} * x{i}"
        coefficients.append(term)

    equation = "y = " + intercept + " " + " ".join(coefficients)

    print("\n最终回归模型:")
    print(equation)

    # 绘制RMSE曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_rmse_history, label='训练集RMSE', linewidth=2)
    plt.plot(test_rmse_history, label='测试集RMSE', linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel('RMSE')
    plt.title('训练集与测试集RMSE随迭代次数变化')
    plt.legend()
    plt.grid(True)

    # 保存图表并显示
    output_file = '梯度下降_RMSE曲线.png'
    plt.savefig(output_file)
    plt.tight_layout()  # 确保布局合理
    plt.show()

    print(f"\nRMSE曲线已保存为: {output_file}")


# 运行梯度下降回归
if __name__ == "__main__":
    run_gradient_descent_regression()