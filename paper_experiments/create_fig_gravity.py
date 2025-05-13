# %% [markdown]
# ---
# title: Training GNN on gravity-like system
# author: Cédric Allier, Michael Innerberger, Stephan Saalfeld
# categories:
#   - Particles
# execute:
#   echo: false
# image: "create_fig_gravity_files/figure-html/cell-gravity-output-1.png"
# ---

# %% [markdown]
# This script generates Supplementary Figure 7.
# A GNN learns the motion rules governing a gravity-like system
# The simulation used to train the GNN consists of 960 particles of 16 different masses.
# The particles interact with each other according to gravity law.

# %%
#| output: false
import os

import umap
import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data

from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.generators import data_generate_particles
from ParticleGraph.models import data_train, data_test
from ParticleGraph.plotting import get_figures, load_and_display
from ParticleGraph.utils import set_device, to_numpy

# %% [markdown]
# First, we load the configuration file and set the device.

# %%
#| echo: true
#| output: false
config_file = 'gravity_16'
figure_id = 'supp7'
config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
device = set_device("auto")

# %% [markdown]
# The following model is used to simulate the gravity-like system with PyTorch Geometric.
#
# %%
#| echo: true
class GravityModel(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the acceleration of particles as a function of their relative position according to the gravity law.

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the acceleration of the particles (dimension 2)
    """

    def __init__(self, aggr_type=[], p=[], clamp=[], pred_limit=[], bc_dpos=[]):
        super(GravityModel, self).__init__(aggr='add')  # "mean" aggregation.

        self.p = p
        self.clamp = clamp
        self.pred_limit = pred_limit
        self.bc_dpos = bc_dpos

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        particle_type = to_numpy(x[:, 5])

        mass = self.p[particle_type]
        dd_pos = self.propagate(edge_index, pos=x[:, 1:3], mass=mass[:, None])
        return dd_pos

    def message(self, pos_i, pos_j, mass_j):
        distance_ij = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1))
        distance_ij = torch.clamp(distance_ij, min=self.clamp)
        direction_ij = self.bc_dpos(pos_j - pos_i) / distance_ij[:, None]
        dd_pos = mass_j * direction_ij / (distance_ij[:, None] ** 2)

        return torch.clamp(dd_pos, max=self.pred_limit)


def bc_pos(x):
    return torch.remainder(x, 1.0)


def bc_dpos(x):
    return torch.remainder(x - 0.5, 1.0) - 0.5

# %% [markdown]
# The training data is generated with the above Pytorch Geometric model
#
# Vizualizations of the particle motions can be found in "decomp-gnn/paper_experiments/graphs_data/gravity_16/"
#
# If the simulation is too large, you can decrease n_particles (multiple of 16) in "gravity_16.yaml"
#
# %%
#| echo: true
#| output: false
p = torch.squeeze(torch.tensor(config.simulation.params))
model = GravityModel(aggr_type=config.graph_model.aggr_type, p=torch.squeeze(p), clamp=config.training.clamp,
              pred_limit=config.training.pred_limit, bc_dpos=bc_dpos)

generate_kwargs = dict(device=device, visualize=True, run_vizualized=0, style='color', alpha=1, erase=True, save=True, step=10)
train_kwargs = dict(device=device, erase=True)
test_kwargs = dict(device=device, visualize=True, style='color', verbose=False, best_model='20', run=0, step=20, save_velocity=True)

data_generate_particles(config, model, bc_pos, bc_dpos, **generate_kwargs)

# %%
#| fig-cap: "Initial configuration of the simulation. There are 960 particles. The colors indicate different masses."
load_and_display('graphs_data/graphs_gravity_16/Fig/Fig_0_0.tif')

# %%
#| fig-cap: "Frame 1800 out of 2000"
load_and_display('graphs_data/graphs_gravity_16/Fig/Fig_0_1800.tif')

# %% [markdown]
# The GNN model (see src/ParticleGraph/models/Interaction_Particle.py) is trained and tested.
#
# Since we ship the trained model with the repository, this step can be skipped if desired.
#
# %%
#| echo: true
#| output: false
if not os.path.exists(f'log/try_{config_file}'):
    data_train(config, config_file, **train_kwargs)

# %% [markdown]
# During training the embedding is saved in
# "paper_experiments/log/try_gravity_16/tmp_training/embedding"
# The plot of the pairwise interactions is saved in
# "paper_experiments/log/try_gravity_16/tmp_training/function"
#
# The model that has been trained in the previous step is used to generate the rollouts.
# %%
#| echo: true
#| output: false
data_test(config, config_file, **test_kwargs)

# %% [markdown]
# Finally, we generate the figures that are shown in Supplementary Figure 7.
# The results of the GNN post-analysis are saved into 'decomp-gnn/paper_experiments/log/try_gravity_16/results'.
# %%
#| echo: true
#| output: false
config_list, epoch_list = get_figures(figure_id, device=device)

# %%
#| fig-cap: "Learned latent vectors (x960)"
load_and_display('log/try_gravity_16/results/first_embedding_gravity_16_20.tif')

# %%
#| fig-cap: "Learned interaction functions (x16)"
load_and_display('log/try_gravity_16/results/func_all_gravity_16_20.tif')

# %%
#| fig-cap: "Learned masses (x16)"
load_and_display('log/try_gravity_16/results/mass_gravity_16.tif')

# %%
#| fig-cap: "GNN rollout inference at frame 1980"
load_and_display('log/try_gravity_16/tmp_recons/Fig_gravity_16_1980.tif')

# %% [markdown]
# All frames can be found in "decomp-gnn/paper_experiments/log/try_gravity_16/tmp_recons/"
# %%
#| echo: true
#| output: false
