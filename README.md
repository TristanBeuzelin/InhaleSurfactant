# InhaleSurfactant

This repository aims at reproducing every figure in **cite paper**. The *requirements.txt* file gives the minimum necessary environment

## Params
YAML parameters file in params/params.yml give the following values for each lung type (infant and adult):

- Surfactant viscosity (Pa.s)
- Surfactant surface tension (N/m)
- Surfactant density (kg/m^3)
- Tracheal radius (m)
- Injection volume (m^3)
- Injection flow rate (m^3/s)
- Number of lung generations

## Scripts

Each script uses *src* code to reproduce a figure :

- **single_injection.py** performs single injections to reproduce Figure 1 plot.
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
creating the surfactant object, then the tree. We set the tree angles (*Tree.set_angles*), generate its geometry (*Tree.generate_tree*), inject surfactant volume (*Tree.inject*), compute the full flow from trachea to final nodes (*Tree.loop*) and compute metrics (*Tree.compute_efficiency* and *Tree.compute_std_inv").
