import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from labellines import labelLine


def f(x, params):
    rho = params["rho"]
    g = 9.81
    a = params["a"]
    L = 6 * a
    lamb = 2 ** (-1 / 3)
    sigma = params["sigma"]
    Q0 = params["flow_rate"]
    U0 = Q0 / (np.pi * a**2)
    V0 = params["V0"]

    k1 = np.exp(-2 * (x**0.523))
    k1_lamb = np.exp(-2 * ((lamb * x) ** 0.523))

    h1 = (1 - 0.36 * (1 - k1)) ** 2
    h1_lamb = (1 - 0.36 * (1 - k1_lamb)) ** 2

    A = rho * g * (a**2) * (lamb**4) / (8 * np.sqrt(2) * U0)
    B = rho * (a**4) * (lamb**2) * np.pi * U0 / (16 * V0)
    C = np.pi * (a**2) * L / V0

    return (
        sigma * x / U0
        - lamb * A / h1_lamb
        - 2 * (lamb**2) * B * h1_lamb / (1 - C * (1 - h1 - h1_lamb))
    )


def f_prim(x, params):
    rho = params["rho"]
    g = 9.81
    a = params["a"]
    L = 6 * a
    lamb = 2 ** (-1 / 3)
    sigma = params["sigma"]
    Q0 = params["flow_rate"]
    U0 = Q0 / (np.pi * a**2)
    V0 = params["V0"]

    k1 = np.exp(-2 * (x**0.523))
    k1_lamb = np.exp(-2 * ((lamb * x) ** 0.523))
    k1_prim = -2 * (x ** (0.523 - 1)) * 0.523 * k1
    k1_lamb_prim = -2 * ((lamb * x) ** (0.523 - 1)) * 0.523 * k1_lamb

    h1 = (1 - 0.36 * (1 - k1)) ** 2
    h1_lamb = (1 - 0.36 * (1 - k1_lamb)) ** 2
    h1_prim = 2 * 0.36 * k1_prim * (1 - 0.36 * (1 - k1))
    h1_lamb_prim = 2 * 0.36 * k1_lamb_prim * (1 - 0.36 * (1 - k1_lamb))

    A = rho * g * (a**2) * (lamb**4) / (8 * np.sqrt(2) * U0)
    B = rho * (a**4) * (lamb**2) * np.pi * U0 / (16 * V0)
    C = np.pi * (a**2) * L / V0

    return (
        sigma / U0
        + lamb**2 * A * h1_lamb_prim / (h1_lamb**2)
        + 2
        * lamb**2
        * B
        * (lamb * h1_lamb_prim * (1 - C * (1 - h1)) - C * h1_lamb * h1_prim)
        / (1 - C * (1 - h1 - h1_lamb) ** 2)
    )


def newton_raphson(x_0, f, f_prim, params, epsilon=1e-8):

    f_eval = f(x_0, params)

    while np.abs(f_eval) > epsilon:
        f_prim_eval = f_prim(x_0, params)
        next_x = x_0 - (f_eval / f_prim_eval)
        f_eval = f(next_x, params)
        x_0 = next_x

    return next_x


def single_newton_raphson():
    """Single Newton-Raphson to find the critical viscosity for the second bifurcation in supine position."""
    TYPE = "adult"
    with open("../params/params.yml", "r") as file:
        params = yaml.safe_load(file)[TYPE]
        params["type"] = TYPE
        params["tree_gamma"] = 0.0
        params["tree_phi"] = 0.0
        params["flow_rate"] = 70e-6
        params["V0"] = 70e-6

    a = params["a"]
    sigma = params["sigma"]
    Q0 = params["flow_rate"]
    U0 = Q0 / (np.pi * a**2)
    x0 = 1.0 * U0 / sigma
    muc = newton_raphson(x0, f, f_prim, params) * sigma / U0
    print(f"Critical viscosity : {muc} Pa.s")


def main():
    """Reproduces Figure 3.b. phase diagram of critical viscosities"""
    TYPE = "adult"
    SAVE_PATH = "./"

    with open("../params/params.yml", "r") as file:
        params = yaml.safe_load(file)[TYPE]
        params["type"] = TYPE
        params["tree_gamma"] = 0.0
        params["tree_phi"] = 0.0
        params["flow_rate"] = 70e-6
        params["V0"] = 70e-6
    a = params["a"]
    sigma = params["sigma"]
    Q0 = params["flow_rate"]
    U0 = Q0 / (np.pi * a**2)
    x0 = 1.0 * U0 / sigma
    V0 = np.linspace(70e-6, 420e-6, 2000)
    Q0 = np.linspace(70e-6, 420e-6, 2000)
    U0 = Q0 / (np.pi * (a**2))
    xs = np.zeros(shape=(V0.shape[0], U0.shape[0]))
    mus = np.zeros(shape=(V0.shape[0], U0.shape[0]))

    for i, V in enumerate(V0):
        for j, Q in enumerate(Q0):
            params["flow_rate"] = Q
            params["V0"] = V
            x = newton_raphson(x0, f, f_prim, params)
            xs[i, j] = x
            U = Q / (np.pi * (a**2))
            mus[i, j] = x * sigma / U

    viscosities = [3e-2, 4e-2, 5e-2, 6e-2, 8e-2, 1.2e-1]
    plt.cla()
    plt.clf()
    fig, ax = plt.subplots()
    fig.set_size_inches(5.8, 5.8)

    for viscosity in viscosities:
        if viscosity == 3e-2:
            threshold = 5 * 1e-5
        elif viscosity == 4e-2:
            threshold = 10 * 1e-5
        elif viscosity == 5e-2:
            threshold = 14 * 1e-5
        elif viscosity == 6e-2:
            threshold = 20 * 1e-5
        elif viscosity == 8e-2:
            threshold = 14 * 3e-5
        elif viscosity == 1e-1:
            threshold = 12 * 5e-4
        else:
            threshold = 2 * 5e-4
        mask = np.abs(mus - viscosity) < threshold
        indices = np.argwhere(mask)
        flow_rates = Q0[indices[:, 1]]
        volumes = V0[indices[:, 0]]
        ax.plot(
            10**6 * flow_rates,
            10**6 * volumes,
            label=f"{viscosity:.2f} Pa.s",
            color="black",
        )
    lines = ax.get_lines()
    fontsize = 15
    for line, viscosity in zip(lines, viscosities):
        if viscosity == 3e-2:
            labelLine(line, 390, rotation=-55, fontsize=fontsize)
        elif viscosity == 4e-2:
            labelLine(
                line, 275, rotation=-65, yoffset=-30, xoffset=10, fontsize=fontsize
            )
        elif viscosity == 5e-2:
            labelLine(line, 215, rotation=-80, yoffset=-10, fontsize=fontsize)
        elif viscosity == 6e-2:
            labelLine(line, 175, rotation=-85, yoffset=-20, fontsize=fontsize)
        elif viscosity == 8e-2:
            labelLine(line, 130, rotation=-90, yoffset=30, fontsize=fontsize)
        elif viscosity == 1e-1:
            labelLine(line, 103, rotation=-90, yoffset=38, fontsize=fontsize)
        else:
            labelLine(line, 85, rotation=-90, yoffset=45, fontsize=fontsize)
    plt.ylabel("Volume (mL)", size=20)
    plt.xlabel("Flow rate (mL/s)", size=20)
    plt.xlim([70, 420])
    plt.ylim([70, 420])
    plt.grid()
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.tick_params(axis="both", which="minor", labelsize=15)

    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (140, 70), 140, 210, edgecolor="green", facecolor="none", alpha=1, linewidth=8
    )
    norm = mpl.colors.Normalize(vmin=mus.min(), vmax=mus.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.plasma)
    cmap.set_array([])
    cmap2 = mpl.colormaps.get_cmap(
        "plasma"
    )  # viridis is the default colormap for imshow
    ax.imshow(mus, cmap=cmap2, origin="lower", extent=[70, 420, 70, 420], alpha=0.5)
    ax.add_patch(rect)

    cbar = fig.colorbar(cmap, ax=ax, alpha=0.5)
    cbar.set_label("Critical viscosity (Pa.s)", size=20)
    cbar.ax.tick_params(labelsize=15)
    plt.savefig(
        f"{SAVE_PATH}/figure_3b.svg",
        dpi=500,
        format="svg",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
