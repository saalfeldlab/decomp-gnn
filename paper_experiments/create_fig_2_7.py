# %% [markdown]
# ---
# title: Signaling system with 998 nodes
# author: Cédric Allier, Michael Innerberger, Stephan Saalfeld
# categories:
#   - Signaling
# execute:
#   echo: false
# image: "create_fig_2_7_files/figure-html/cell-10-output-1.png"
# ---

# %% [markdown]
# This script creates the seventh column of paper's Figure 2.
# Simulation of a signaling network, 986 nodes, 17,865 edges, 2 types of nodes.
# Note 100 of datasets are generated to test training with multiple trials.

# %%
#| output: false
import os

import umap
import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data

from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.generators import data_generate_synaptic
from ParticleGraph.models import data_train, data_test
from ParticleGraph.plotting import get_figures, load_and_display
from ParticleGraph.utils import set_device, to_numpy

# %% [markdown]
# First, we load the configuration file and set the device.

# %%
#| echo: true
#| output: false
config_file = 'signal_N_100_2'
config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
device = set_device("auto")

# %% [markdown]
# The following model is used to simulate the signaling network with PyTorch Geometric.
#
# %%
#| echo: true

class SignalingNetwork(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float

    """

    def __init__(self, aggr_type=[], p=[], bc_dpos=[]):
        super(SignalingNetwork, self).__init__(aggr=aggr_type)

        self.p = p
        self.bc_dpos = bc_dpos

    def forward(self, data=[], return_all=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        particle_type = x[:, 5].long()
        parameters = self.p[particle_type]
        b = parameters[:, 0:1]
        c = parameters[:, 1:2]

        u = x[:, 6:7]

        msg = self.propagate(edge_index, u=u, edge_attr=edge_attr)

        du = -b * u + c * torch.tanh(u) + msg

        if return_all:
            return du, -b * u + c * torch.tanh(u), msg
        else:
            return du

    def message(self, u_j, edge_attr):

        self.activation = torch.tanh(u_j)
        self.u_j = u_j

        return edge_attr[:, None] * torch.tanh(u_j)


def bc_pos(x):
    return torch.remainder(x, 1.0)


def bc_dpos(x):
    return torch.remainder(x - 0.5, 1.0) - 0.5

# %% [markdown]
# The data is generated with the above Pytorch Geometric model.
# If the simulation is too large, you can decrease n_particles (multiple of 2) and n_nodes in "signal_N_100_2.yaml"
#
# %%
#| echo: true
#| output: false
p = torch.squeeze(torch.tensor(config.simulation.params))
model = SignalingNetwork(aggr_type=config.graph_model.aggr_type, p=torch.squeeze(p), bc_dpos=bc_dpos)

generate_kwargs = dict(device=device, visualize=True, run_vizualized=0, style='color', alpha=1, erase=True, save=True, step=10)
train_kwargs = dict(device=device, erase=True)
test_kwargs = dict(device=device, visualize=True, style='color', verbose=False, best_model='20', run=0, step=1, save_velocity=True)

data_generate_synaptic(config, model, **generate_kwargs)


# %% [markdown]
# Finally, we generate the figures that are shown in Figure 2.
# The frames of the first six datasets are saved in 'decomp-gnn/paper_experiments/graphs_data/graphs_signal_N_100_2/Fig/'.
# %%
#| echo: true
#| output: false

# %%
#| fig-cap: "Initial configuration of the simulation. There are 998 nodes. The colors indicate the node scalar values."
load_and_display('graphs_data/graphs_signal_N_100_2/Fig/Fig_0_10000.tif')

# %%
#| fig-cap: "Frame 300 out of 1000"
load_and_display('graphs_data/graphs_signal_N_100_2/Fig/Fig_0_10250.tif')

# %%
#| fig-cap: "Frame 600 out of 1000"
load_and_display('graphs_data/graphs_signal_N_100_2/Fig/Fig_0_10500.tif')

# %%
#| fig-cap: "Frame 900 out of 1000"
load_and_display('graphs_data/graphs_signal_N_100_2/Fig/Fig_0_10750.tif')

