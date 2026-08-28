import matplotlib.pyplot as plt
import numpy as np
import yaml
from src.model import Tree
from src.surfactant import Surfactant


def main():
    """Reproduces Figure 3.a plot."""

    TYPE = "adult"
    SAVE_PATH = "./"

    with open("../params/params.yml", "r") as file:
        params = yaml.safe_load(file)[TYPE]
        params["type"] = TYPE
        params["tree_gamma"] = 0.0
        params["tree_phi"] = 0.0

    mus = np.linspace(1e-2, 1, 200)
    volumes = [
        70e-6,
        100e-6,
        150e-6,
        250e-6,
        350e-6,
        400e-6,
    ]
    flow_rate = 100e-6
    volume = 200e-6

    nb_alim_nodes = []
    N_tot = 2 ** params["n_gen"]

    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(4.8)

    for i, volume in enumerate(volumes):
        nb_alim_nodes = []
        for mu in mus:
            surf = Surfactant(mu=mu, sigma=params["sigma"], rho=params["rho"])
            tree = Tree(
                surfactant=surf,
                a0=params["a"],
                n_gen=params["n_gen"],
                track_ruptures=True,
                coating=True,
            )

            tree.set_angles(gamma=params["tree_gamma"], phi=params["tree_phi"])
            tree.generate_tree()
            tree.inject(volume=volume, flow_rate=flow_rate)
            tree.loop()
            tree.compute_efficiency()
            tree.compute_std_inv()

            nb_alim_nodes.append(tree.final_volumes[tree.final_volumes != 0.0].shape[0])
        if i >= 7:
            ax.plot(
                mus,
                nb_alim_nodes,
                c="black",
                label=f"{volume * 10**6:.0f}  - {volumes[-1] * 10**6:.0f} mL",
                linewidth=4,
            )
        elif i == 0:
            ax.plot(
                mus,
                nb_alim_nodes,
                c="blue",
                label=f"{volume * 10**6:.0f}  - {volumes[-1] * 10**6:.0f} mL",
                linewidth=4,
            )
        else:
            ax.plot(
                mus,
                nb_alim_nodes,
                c="black",
                label=f"{volume * 10**6:.0f} mL",
                linewidth=4,
            )

    plt.ylabel("Number of supplied final nodes", size=20)
    plt.xlabel("Viscosity (Pa.s)", size=20)
    plt.yticks([0, N_tot / 4, N_tot / 2, N_tot * 3 / 4, N_tot])
    plt.xlim([0.01, 1])
    plt.ylim([0, 2 ** (params["n_gen"])])
    plt.xscale("log")
    plt.xticks([0.01, 0.1, 1])
    plt.tick_params(axis="both", which="major", labelsize=15)
    plt.tick_params(axis="both", which="minor", labelsize=15)
    plt.tick_params(axis='x', which='both', labelsize=17)
    plt.axvline(x=0.103, color="r", linewidth=4)
    plt.axvline(x=0.03, color="g", linewidth=4)
    plt.grid(visible=True)
    plt.savefig(f"{SAVE_PATH}/figure_3a.svg", dpi=500, format="svg")


if __name__ == "__main__":
    main()
