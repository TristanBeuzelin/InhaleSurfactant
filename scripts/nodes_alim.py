import yaml
import numpy as np
import matplotlib.pyplot as plt
from src.surfactant import Surfactant
from src.model import Tree
from labellines import labelLines

def main():
    """Reproduces Figure 3.a plot."""

    TYPE = "adult"
    SAVE_PATH = "./"

    with open('../params/params.yml', 'r') as file:
        params = yaml.safe_load(file)[TYPE]
        params["type"] = TYPE
        params["tree_gamma"] = 0.0
        params["tree_phi"] = 0.0

    mus = np.linspace(1e-2, 1, 200)
    volumes = [70e-6, 150e-6, 250e-6, 350e-6, 400e-6]
    flow_rates = np.linspace(70e-6, 420e-6, 10)
    flow_rate = 70e-6
    volume = 200e-6

    nb_alim_nodes = []
    N_tot = 2 ** params["n_gen"]

    fig, ax = plt.subplots()
    fig.set_figwidth(10)
    fig.set_figheight(4.8)

    for i, volume in enumerate(volumes):
        nb_alim_nodes = []
        flag_second_embranchement = False
        for mu in mus:
            surf = Surfactant(
                    mu = mu,
                    sigma = params["sigma"],
                    rho = params["rho"]
            )
            tree = Tree(
                surfactant=surf,
                a0=params["a"],
                n_gen=params["n_gen"],
                track_ruptures=True,
                coating=True
            )
            tree.set_angles(
                gamma=params["tree_gamma"],
                phi=params["tree_phi"]
            )
            tree.generate_tree()
            tree.inject(volume=volume, flow_rate=flow_rate)
            tree.loop()
            tree.compute_efficiency()
            tree.compute_std_inv()
            
            if (tree.alphas[1] != 0.0).any() and not flag_second_embranchement:
                print(f"Critical viscosity for second embranchement : {mu:.5f} Pa.s")
                flag_second_embranchement = True
            nb_alim_nodes.append(tree.final_volumes[tree.final_volumes != 0.0].shape[0])

        surf = Surfactant(
                    mu = 3e-2,
                    sigma = params["sigma"],
                    rho = params["rho"]
            )
        tree = Tree(
            surfactant=surf,
            a0=params["a"],
            n_gen=params["n_gen"],
            track_ruptures=True,
            coating=True
        )
        tree.set_angles(
            gamma=params["tree_gamma"],
            phi=params["tree_phi"]
        )
        tree.generate_tree()
        tree.inject(volume=volume, flow_rate=flow_rate)
        tree.loop()
        tree.compute_efficiency()
        tree.compute_std_inv()
        print(f"Efficiency : {tree.efficiency} %")
        print(f"1/Std : {tree.std_inv}")
        N_curosurf = tree.final_volumes[tree.final_volumes != 0.0].shape[0]

        if i >= 7:
            ax.plot(mus, nb_alim_nodes, c="black", label=f"{volume * 10 ** 6:.0f}  - {volumes[-1] * 10 ** 6:.0f} mL", linewidth=2)
        else:
            ax.plot(mus, nb_alim_nodes, c="black", label=f"{volume * 10 ** 6:.0f} mL", linewidth=2)
    xvals = [0.5, 0.45, 0.35, 0.3, 0.25, 0.20, 0.15]
    labelLines(ax.get_lines(), zorder=2.5, xvals=xvals, fontsize=15)

    plt.ylabel("Number of supplied final nodes", size=20)
    plt.xlabel("Viscosity (Pa.s)", size=20)
    plt.xticks([3e-2, 0.146, 3e-1, 1])
    plt.yticks([0, N_tot / 4, N_tot / 2, N_tot * 3 / 4, N_tot])
    plt.xlim([0, 1])
    plt.ylim([0, 2 ** (params["n_gen"])])
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tick_params(axis='both', which='minor', labelsize=15)
    plt.axvline(x=0.146, color="r")
    plt.grid(visible=True)
    plt.savefig("/".join((SAVE_PATH, "nodes_alim_mus.svg")), dpi=500, format="svg")

if __name__ == "__main__":
    main()