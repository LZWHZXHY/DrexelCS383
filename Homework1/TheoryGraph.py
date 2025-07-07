import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot_j_vs_x1():

    matplotlib.use('TkAgg')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12

    def J(x1, x2):
        return (x1 + x2 - 2) ** 2

    x1 = np.linspace(-5, 5, 500)

    fig, ax = plt.subplots()

    colors = ['blue', 'orange', 'green']
    for i, x2_val in enumerate([0, 1, 2]):
        j_vals = J(x1, x2_val)
        ax.plot(x1, j_vals,
                label=f'$x_2 = {x2_val}$',
                color=colors[i],
                linewidth=2)

        min_x1 = 2 - x2_val
        min_J = J(min_x1, x2_val)
        ax.scatter(min_x1, min_J,
                   color='red',
                   s=100,
                   zorder=5,
                   edgecolor='black',
                   label=f'Min at $x_1={min_x1}$' if i == 0 else None)


    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$J = (x_1 + x_2 - 2)^2$', fontsize=14)
    ax.set_title('$J$ vs $x_1$ for Fixed $x_2$ Values', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()


    ax.text(0.05, 0.95,
            r'$J = (x_1 + x_2 - 2)^2$',
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top')


    ax.text(0.05, 0.85,
            'Minimum when $x_1 + x_2 = 2$',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top')


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_j_vs_x1()