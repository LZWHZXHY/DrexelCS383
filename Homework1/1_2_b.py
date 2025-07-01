import numpy as np
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt


def J(x1, x2):
    return (x1 + x2 - 2) ** 2


x1 = np.linspace(-5, 5, 100)
plt.figure(figsize=(10, 6))

# 为每个固定 x₂ 值绘制曲线
for x2_val in [0, 1, 2]:
    j_vals = J(x1, x2_val)
    plt.plot(x1, j_vals, label=f'$x_2 = {x2_val}$')

    # 标记最小值点
    min_x1 = 2 - x2_val
    min_J = J(min_x1, x2_val)
    plt.scatter(min_x1, min_J, color='red', s=50, zorder=5)

plt.xlabel('$x_1$', fontsize=12)
plt.ylabel('$J = (x_1 + x_2 - 2)^2$', fontsize=12)
plt.title('$J$ vs $x_1$ for Fixed $x_2$ Values', fontsize=14)
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存图像而不是显示
plt.savefig('x1_vs_J_plot.png', dpi=300)
print("Plot saved as 'x1_vs_J_plot.png'")