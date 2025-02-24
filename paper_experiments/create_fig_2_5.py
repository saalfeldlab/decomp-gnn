# %% [markdown]
# <!--
# title: Attraction-repulsion system with 3 particle types
# author: Cédric Allier, Michael Innerberger, Stephan Saalfeld
# categories:
#   - Particles
# execute:
#   echo: false
# image: "create_fig_2_5_files/figure-html/cell-10-output-1.png"
# -->

# %% [markdown]
# This script creates the fifth column of paper's Figure 2.
# Simulation of wave-propagation over a mesh of 1E4 nodes with
# variable propagation-coefficients.

# %%
#| output: false
import os

import umap
import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data

from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.generators import data_generate_mesh
from ParticleGraph.models import data_train, data_test
from ParticleGraph.plotting import get_figures, load_and_display
from ParticleGraph.utils import set_device, to_numpy
from tifffile import imread
import numpy as np

# %% [markdown]
# First, we load the configuration file and set the device.

# %%
#| echo: true
#| output: false
config_file = 'wave_slit_ter'
config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
device = set_device("auto")

# %% [markdown]
# The following model is used to simulate the wave propagation with PyTorch Geometric.
#
# %%
#| echo: true


class WaveModel(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the Laplacian of a scalar field.

    Inputs
    ----------
    data : a torch_geometric.data object
    note the Laplacian coeeficients are in data.edge_attr

    Returns
    -------
    laplacian : float
        the Laplacian
    """

    def __init__(self, aggr_type=[], beta=[], bc_dpos=[]):
        super(WaveModel, self).__init__(aggr='add')  # "mean" aggregation.

        self.beta = beta
        self.bc_dpos = bc_dpos
        self.coeff = []

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # if self.coeff == []:
        #     particle_type = to_numpy(x[:, 5])
        #     c = self.c[particle_type]
        #     c = c[:, None]
        # else:

        c = self.coeff
        u = x[:, 6:7]

        laplacian_u = self.propagate(edge_index, u=u, edge_attr=edge_attr)
        dd_u = self.beta * c * laplacian_u

        self.laplacian_u = laplacian_u

        return dd_u

        pos = to_numpy(data.x)
        deg = pyg_utils.degree(edge_index[0], data.num_nodes)
        plt.ion()
        plt.scatter(pos[:,1],pos[:,2], s=20, c=to_numpy(deg),vmin=7,vmax=10)

    def message(self, u_j, edge_attr):
        L = edge_attr[:,None] * u_j

        return L

def bc_pos(x):
    return torch.remainder(x, 1.0)


def bc_dpos(x):
    return torch.remainder(x - 0.5, 1.0) - 0.5

# %% [markdown]
# The data is generated with the above Pytorch Geometric model.
# Note two datasets are generated, one for training and one for validation.
#
# %%
#| echo: true
#| output: false

model = WaveModel(aggr_type=config.graph_model.aggr_type, beta=config.simulation.beta)

generate_kwargs = dict(device=device, visualize=True, run_vizualized=0, style='color', erase=False, save=True, step=10)
train_kwargs = dict(device=device, erase=True)
test_kwargs = dict(device=device, visualize=True, style='color', verbose=False, best_model='20', run=0, step=1, save_velocity=True)

data_generate_mesh(config, model , **generate_kwargs)


# %% [markdown]
# Finally, we generate the figures that are shown in Figure 2.
# %%
#| echo: true
#| output: false

# %%
#| fig-cap: "Initial configuration of the simulation. There are 1E4 nodes. The colors indicate the node scalar values."
load_and_display('graphs_data/graphs_wave_slit_ter/Fig/Fig_0_0.tif')

# %%
#| fig-cap: "Frame 2500 out of 8000"
load_and_display('graphs_data/graphs_wave_slit_ter/Fig/Fig_0_2500.tif')

# %%
#| fig-cap: "Frame 5000 out of 8000"
load_and_display('graphs_data/graphs_wave_slit_ter/Fig/Fig_0_5000.tif')

# %%
#| fig-cap: "Frame 7500 out of 8000"
load_and_display('graphs_data/graphs_wave_slit_ter/Fig/Fig_0_7500.tif')

