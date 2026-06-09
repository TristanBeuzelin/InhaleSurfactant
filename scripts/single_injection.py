import datetime

import h5py
import numpy as np
import yaml

from src.model import Tree
from src.plot_tree import Tree_plot
from src.surfactant import Surfactant


def main(figure, viscosity, type):
    """
    Reproduce Figure 1 plots by performing a simple injection simulation.
    Results are saved in h5 format.
    """
    TYPE = type
    SAVE_PATH = "./"
    PLOT_3D = True

    with open("../params/params.yml", "r") as file:
        params = yaml.safe_load(file)[TYPE]

    params["type"] = TYPE
    params["tree_gamma"] = 0.0
    params["tree_phi"] = 0.0

    surf = Surfactant(
        mu=viscosity,
        sigma=params["sigma"],
        rho=params["rho"],
    )

    tree = Tree(
        surfactant=surf,
        a0=params["a"],
        n_gen=params["n_gen"],
        track_ruptures=True,
        coating=True,
    )
    tree.set_angles(
        gamma=params["tree_gamma"],
        phi=params["tree_phi"],
    )
    tree.generate_tree()
    tree.inject(volume=params["V0"], flow_rate=params["flow_rate"])
    tree.loop()
    tree.compute_efficiency()
    tree.compute_std_inv()

    print("Injection done, metrics:")
    print(f"  Efficiency : {tree.efficiency} %")
    print(f"  1/Std      : {tree.std_inv}\n")

    final_positions = tree.full_vecs[-1]
    data = np.concatenate(
        (final_positions, tree.homogeneity.reshape((-1, 1))),
        axis=1,
    )

    output_path = f"{SAVE_PATH}/inj_0.h5"
    with h5py.File(output_path, "w") as f:
        f.attrs["created_at"] = datetime.datetime.now().isoformat()

        inj_params = f.create_group("injection_parameters")
        inj_params.attrs.update(
            {
                "type": TYPE,
                "gamma": params["tree_gamma"],
                "phi": params["tree_phi"],
                "viscosity": params["mu"],
                "surface_tension": params["sigma"],
                "density": params["rho"],
                "a0": params["a"],
                "n_gen": params["n_gen"],
                "coating": tree.coating,
                "volume": params["V0"],
                "flow_rate": params["flow_rate"],
            }
        )

        res = f.create_group("results")
        res.create_dataset("final_volumes", data=data)
        res.attrs["Efficiency"] = tree.efficiency
        res.attrs["Homogeneity"] = tree.std_inv

    if PLOT_3D:
        first_vec = np.array(
            (
                0.0,
                -np.cos(params["tree_gamma"]),
                -np.sin(params["tree_gamma"]),
            )
        )
        tree_plt = Tree_plot(
            n_gen=params["n_gen"],
            first_vec=first_vec,
            phi=params["tree_phi"],
            gamma=params["tree_gamma"],
        )
        tree_plt.loop()
        tree_plt.plot_end(
            volumes=tree.homogeneity,
            path=SAVE_PATH,
            filename=f"figure_1{figure}.png",
        )


if __name__ == "__main__":
    main()
