# Decomposing heterogeneous dynamical systems with graph neural networks
Cédric Allier, Magdalena C. Schneider, Michael Innerberger, Larissa Heinrich, John A. Bogovic, Stephan Saalfeld

Janelia Research Campus, Howard Hughes Medical Institute

![Summary of the GNN workflow](ressources/gnn-summary.png)

### Setup
Run the following line from the terminal to create a new environment 'gnn':
```
conda env create -f environment.yaml
```

Activate the environment:
```
conda activate gnn
```

Install the ParticleGraph package by executing the following command from the root of this directory:
```
pip install -e .
```

Then, you should be able to import all the modules from the package in python:
```python
from ParticleGraph import *
```

To create the paper's figures run the scripts in paper_experiments.

This create folder of data in paper_experimets/graphs_data.

Time-lapse movies can be found in Fig folders.


### Citation
```
@article{allier2024decompgnn,
  author = {Allier, Cédric
            and Schneider, Magdalena C.
            and Innerberger, Michael
            and Heinrich, Larissa
            and Bogovic, John A.
            and Saalfeld Stephan},
  title = {Decomposing heterogeneous dynamical systems with graph neural networks},
  archiveprefix = {arXiv},
  eprint = {2102.04360},
  year = {2024}
}
```
