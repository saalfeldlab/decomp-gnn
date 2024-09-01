# decomp-gnn
Decomposing heterogeneous dynamical systems with graph neural networks

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
