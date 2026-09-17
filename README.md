# InhaleSurfactant

This repository aims at reproducing every figure in **https://arxiv.org/abs/2609.18450**. It also permits to perform custom simulations of exogenous pulmonary surfactant injections.

## Installation and execution

Using uv:

```bash
git clone https://github.com/TristanBeuzelin/InhaleSurfactant.git
cd InhaleSurfactant
uv sync
```

If uv cannot be used, a *requirements.txt* file is available.

Execute any script with :

```bash
uv run python scripts/script_name.py
```

A *results/* folder is created with all the figures and a simulation *.h5* datafile inside.

## Params
Default YAML parameters file in params/params.yml give the following values for each lung type (infant and adult) used in the paper:

- Surfactant viscosity ($\mathrm{Pa} \cdot \mathrm{s}$)
- Surfactant surface tension ($\mathrm{N} \cdot \mathrm{m}^{-1}$)
- Surfactant density ($\mathrm{kg} \cdot \mathrm{m}^{-3}$)
- Tracheal radius ($\mathrm{m}$)
- Injection volume ($\mathrm{m}^3$)
- Injection flow rate ($\mathrm{m}^3 \cdot \mathrm{s}^{-1}$)
- Number of lung generations

Custom parameters can be loaded using --param.

## Scripts

Each script uses *src* code to reproduce a figure :

- **single_injection.py** performs single injections to reproduce Figure 1 plot. Custom parameter file can be used and loaded with --param argument. Surfactant viscosity and patient type can be quickly tuned using --viscosity and --type args.
- **splitting_factors.py** reproduces Figure 2 plot.
- **nodes_alim.py** reproduces Figure 3.a plot.
- **newton_raphson.py** reproduces Figure 3.b plot.
- **reproduce_all.py** runs all scripts to produce all figures.

### Source code

- **quaternion.py** defines a simple quaternion object with all necessary properties and operations. Quaternions are used in 3D rotations to generate the lung geometry.
- **surfactant.py** defines the surfactant rheology.
- **plot_tree.py** is used in **single_injection.py** to plot the lung perspective.

#### model.py

The model described in (ref article) is represented by the *Tree* class. Each class method is described inside the code. To run a full surfactant injection, we begin by
creating the surfactant object, then the tree. We set the tree angles (*Tree.set_angles*), generate its geometry (*Tree.generate_tree*), inject surfactant volume (*Tree.inject*), compute the full flow from trachea to final nodes (*Tree.loop*) and compute metrics (*Tree.compute_efficiency* and *Tree.compute_std_inv*).
