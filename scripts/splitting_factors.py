import yaml
import numpy as np
import matplotlib.pyplot as plt
from src.surfactant import Surfactant
from src.model import Tree
from src.plot_tree import Tree_plot

def main():
    """Reproduces Figure 2 splitting factors distributions across generations."""

    TYPE = "infant"
    SAVE_PATH = "./"

    with open('../params/params.yml', 'r') as file:
        params = yaml.safe_load(file)[TYPE]
        params["type"] = TYPE
        # Choose orientation
        params["tree_gamma"] = 0.
        params["tree_phi"] = 0.

    fig, ax = plt.subplots()
    colors = ["cornflowerblue", "gold", "red"]
    for mu, c in zip([3e-2, 0.3, 1.0], colors):
        print(mu)
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
        tree.inject(volume=params["V0"], flow_rate=params["flow_rate"])
        tree.loop()
        tree.compute_efficiency()
        tree.compute_std_inv()
        print(f"Efficiency : {tree.efficiency} %")
        print(f"1/Std : {tree.std_inv}")
        alphas = []
        for i, (arr, filt) in enumerate(zip(tree.alphas, tree.split_ruptures)):
            filt = (tree.split_ruptures[i] != -1)
            alphas.append(arr[filt])
        lines = []
        for level, X in enumerate(alphas):
            plt.scatter(x=np.ones(shape=X.shape) * level, y=X, c=c)
            if level > 0:
                for p_alpha, alpha_left in zip(prev_alphas, X[::2]):
                    lines.append((level - 1,p_alpha))
                    lines.append((level, alpha_left))
                    plt.plot((level - 1, level), (p_alpha, alpha_left), color=c, alpha=0.1)
                for p_alpha, alpha_right in zip(prev_alphas, X[1::2]):
                    lines.append((level - 1,p_alpha))
                    lines.append((level, alpha_right))
                    plt.plot((level - 1, level), (p_alpha, alpha_right), color=c, alpha=0.1)
            prev_alphas = X
    plt.ylabel("Splitting factors", size=20)
    plt.xlabel("Generation", size=20)
    plt.ylim([0.0, 0.5])
    ax.yaxis.set_inverted(False)
    ax.set_xticks([i for i in range(params["n_gen"])])
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='both', which='minor', labelsize=20)
    plt.grid(visible=True)
    plt.savefig("".join((SAVE_PATH, f"{params["type"]}_alphas.svg")), dpi=500, format="svg")

if __name__ == "__main__":
    main()