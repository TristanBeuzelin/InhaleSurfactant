import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
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

    _fig, ax = plt.subplots()
    colors = ["cornflowerblue", "gold", "red"]
    for mu, c in zip([3e-2, 0.3, 1.0], colors):
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
        alphas = []
        for i, (arr, filt) in enumerate(zip(tree.alphas, tree.split_ruptures)):
            filt = tree.split_ruptures[i] != -1
            alphas.append(arr[filt])
        for level, X in enumerate(alphas):
            plt.scatter(y=np.ones(shape=X.shape) * level, x=X, c=c)
            if level == 0:
                prev_alphas = X
            else:
                for p_alpha, alpha_left in zip(prev_alphas, X[::2]):
                    plt.plot(
                        (p_alpha, alpha_left), (level - 1, level), color=c, alpha=0.1
                    )
                for p_alpha, alpha_right in zip(prev_alphas, X[1::2]):
                    plt.plot(
                        (p_alpha, alpha_right), (level - 1, level), color=c, alpha=0.1
                    )
                prev_alphas = X
    plt.xlabel("Splitting factors", size=20)
    plt.ylabel("Generation", size=20)
    plt.xlim([0.0, 0.5])
    ax.yaxis.set_inverted(True)
    ax.set_yticks([i for i in range(params["n_gen"])])
    ax.tick_params(axis="both", which="major", labelsize=20)
    ax.tick_params(axis="both", which="minor", labelsize=20)
    plt.grid(visible=True)
    if type == "infant":
        plt.savefig(f"{SAVE_PATH}/figure_2a.svg", dpi=500, format="svg")
    else:
        plt.savefig(f"{SAVE_PATH}/figure_2b.svg", dpi=500, format="svg")


if __name__ == "__main__":
    main()
