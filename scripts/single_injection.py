import yaml
import numpy as np
from src.surfactant import Surfactant
from src.model import Tree
from src.plot_tree import Tree_plot

def main():
    """Reproduces Figure 1 plots by performing a simple injection simulation. Results are saved in h5 format."""

    TYPE = "adult"
    SAVE_PATH = "./"
    PLOT_3D = True

    with open('../params/params.yml', 'r') as file:
        params = yaml.safe_load(file)[TYPE]
        params["type"] = TYPE
        # Choose orientation
        params["tree_gamma"] = 0.
        params["tree_phi"] = 0.

    surf = Surfactant(
            mu = params["mu"],
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
    print("Injection done, metrics :")
    print(f"Efficiency : {tree.efficiency} %")
    print(f"1/Std : {tree.std_inv}\n")

    final_positions = tree.full_vecs[-1]
    final_volumes = tree.final_volumes.reshape((-1, 1))

    data = np.concatenate((final_positions, tree.homogeneity.reshape((-1, 1))), axis=1)

    import h5py
    import datetime
    with h5py.File("/".join((SAVE_PATH, "inj_0.h5")), "w") as f:
        f.attrs["created_at"] = datetime.datetime.now().isoformat()
        inj_params = f.create_group('injection_parameters')
        inj_params.attrs["type"] = TYPE
        inj_params.attrs["gamma"] = params["tree_gamma"]
        inj_params.attrs["phi"] = params["tree_phi"]
        inj_params.attrs["viscosity"] = params["mu"]
        inj_params.attrs["surface_tension"] = params["sigma"]
        inj_params.attrs["density"] = params["rho"]
        inj_params.attrs["a0"] = params["a"]
        inj_params.attrs["n_gen"] = params["n_gen"]
        inj_params.attrs["coating"] = tree.coating
        inj_params.attrs["volume"] = params["V0"]
        inj_params.attrs["flow_rate"] = params["flow_rate"]
        
        res = f.create_group("results")
        res.create_dataset("final_volumes", data=data)
        res.attrs["Efficiency"] = tree.efficiency
        res.attrs["Homogeneity"] = tree.std_inv

    if PLOT_3D:
        tree_plt = Tree_plot(n_gen=params["n_gen"], first_vec=np.array((0, -np.cos(params["tree_gamma"]), -np.sin(params["tree_gamma"]))), phi=params["tree_phi"], gamma=params["tree_gamma"])
        tree_plt.loop()
        tree_plt.plot_end(volumes=tree.homogeneity, path=SAVE_PATH, filename="final_distribution.png")

if __name__ == "__main__":
    main()