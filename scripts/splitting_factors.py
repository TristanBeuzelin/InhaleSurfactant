import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from src.draw_flattened_tree import plot_flattened_tree
from src.model import Tree
from src.surfactant import Surfactant


def main(argv=None):
    """Reproduces Figure 2 splitting factors distributions across generations."""

    REPO_ROOT = Path(__file__).resolve().parent.parent
    SAVE_PATH = Path(f"{REPO_ROOT}/results")
    SAVE_PATH.mkdir(exist_ok=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--param",
        type=Path,
        default=REPO_ROOT / "params/params.yml",
        help="Path to YML parameters file"
    )
    parser.add_argument(
            "--type",
            type=str,
            default="adult",
            help="Patient type"
        )
    args = parser.parse_args(argv)
    type = args.type

    with open(args.param, "r") as file:
        params = yaml.safe_load(file)[type]
        params["type"] = type
        # Choose orientation
        params["tree_gamma"] = 0.0
        params["tree_phi"] = 0.0

    if type == "infant":
        first_plot_name = "figure_2a"
        second_plot_names = ["figure_2c", "figure_2d", "figure_2e"]
    elif type == "adult":
        first_plot_name = "figure_2b"
        second_plot_names = ["figure_2f", "figure_2g", "figure_2h"]
    else:
        raise ValueError("Patient type must be infant or adult.")
    
    fig, ax = plt.subplots()
    viscosities = [3e-2, 0.3, 1.0]
    colors = ["cornflowerblue", "gold", "red"]
    transparencies = [0.3, 0.5, 0.7]

    for mu, c, t, flattened_tree_name in zip(viscosities, colors, transparencies, second_plot_names):
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
        tree.inject(volume=params["V0"], flow_rate=params["flow_rate"])
        tree.loop()
        tree.compute_efficiency()
        tree.compute_std_inv()

        fig2, ax2 = plt.subplots(figsize=(11, 11))
        plot_flattened_tree(
            fig=fig2,
            ax=ax2,
            alphas=tree.alphas,
            save_path=SAVE_PATH,
            file_name=flattened_tree_name
            )

        alphas = []
        for i, (arr, filt) in enumerate(zip(tree.alphas, tree.split_ruptures)):
            filt = tree.split_ruptures[i] != -1
            alphas.append(arr[filt])
        for level, X in enumerate(alphas):
            ax.scatter(y=np.ones(shape=X.shape) * level, x=X, c=c)
        mins = [np.min(X) for X in alphas]
        ax.fill_betweenx(y=[i for i in range(params["n_gen"])], x1=mins, x2=0.5, color=c, alpha=t)
    ax.set_xlabel("Splitting factors", size=20)
    ax.set_ylabel("Generation", size=20)
    ax.set_xlim([0.0, 0.5])
    ax.yaxis.set_inverted(True)
    ax.set_yticks([i for i in range(params["n_gen"])])
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.tick_params(axis="both", which="minor", labelsize=20)
    ax.grid(visible=True)
    fig.savefig(f"{SAVE_PATH}/{first_plot_name}.svg", dpi=500, format="svg")


if __name__ == "__main__":
    main()
