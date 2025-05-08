# %% [markdown]
# ---
# title: Attraction-repulsion system with 3 particle types
# author: Cédric Allier, Michael Innerberger, Stephan Saalfeld
# categories:
#   - Particles
# execute:
#   echo: false
# image: "create_fig_3_1_files/figure-html/cell-12-output-1.png"
# ---

# %% [markdown]
# This script creates the figure supplementary 17.
# A GNN learns the rules governing a reaction-diffusion system.
# The simulation used to train the GNN consists of 10000 nodes of four different types.
# The nodes on a mesh interact with each other according to rock-paper-scissor laws.

# %%
#| output: false
import os

import umap
import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
from tifffile import imread, imsave
import numpy as np

from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.generators import data_generate_mesh
from ParticleGraph.generators import data_generate_particles
from ParticleGraph.generators import init_mesh
from ParticleGraph.models import data_train, data_test
from ParticleGraph.plotting import get_figures, load_and_display
from ParticleGraph.utils import set_device, to_numpy

# %% [markdown]
# First, we load the configuration file and set the device.

# %%
#| echo: true
#| output: false
config_file = 'RD_RPS'
figure_id = 'supplementary_17'
config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
device = set_device("auto")

# %% [markdown]
# The following model is used to simulate the 'rock-paper-scissor' model with PyTorch Geometric.
#
# %%
#| echo: true

class RDModel(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the reaction diffusion according to the rock paper scissor model.

    Inputs
    ----------
    data : a torch_geometric.data object
    Note the Laplacian coeeficients are in data.edge_attr

    Returns
    -------
    increment : float
        the first derivative of three scalar fields u, v and w

    """

    def __init__(self, aggr_type=[], bc_dpos=[], coeff = []):
        super(RDModel, self).__init__(aggr='add')  # "mean" aggregation.

        self.bc_dpos = bc_dpos
        self.coeff = coeff
        self.a = 0.6

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        c = self.coeff

        uvw = data.x[:, 6:9]
        laplace_uvw = c * self.propagate(data.edge_index, uvw=uvw, discrete_laplacian=data.edge_attr)
        p = torch.sum(uvw, axis=1)

        d_uvw = laplace_uvw + uvw * (1 - p[:, None] - self.a * uvw[:, [1, 2, 0]])
        # This is equivalent to the nonlinear reaction diffusion equation:
        #   du = D * laplace_u + u * (1 - p - a * v)
        #   dv = D * laplace_v + v * (1 - p - a * w)
        #   dw = D * laplace_w + w * (1 - p - a * u)

        return d_uvw

    def message(self, uvw_j, discrete_laplacian):
        return discrete_laplacian[:, None] * uvw_j


def bc_pos(x):
    return torch.remainder(x, 1.0)


def bc_dpos(x):
    return torch.remainder(x - 0.5, 1.0) - 0.5

# %% [markdown]
# The coefficients of diffusion are loaded from a tif file specified in the config yaml file and the data is generated.
#
# %%
#| echo: true
#| output: false



X1_mesh, V1_mesh, T1_mesh, H1_mesh, A1_mesh, N1_mesh, mesh_data = init_mesh(config, device=device)

i0 = imread(f'../ressources/{config.simulation.node_coeff_map}')
i0 = np.flipud(i0)
values = i0[(to_numpy(X1_mesh[:, 1]) * 255).astype(int), (to_numpy(X1_mesh[:, 0]) * 255).astype(int)]
values = np.reshape(values, len(X1_mesh))
values = torch.tensor(values, device=device, dtype=torch.float32)[:, None]

model = RDModel(
    aggr_type='add',
    bc_dpos=bc_dpos,
    coeff=values)

generate_kwargs = dict(device=device, visualize=True, run_vizualized=0, style='color', alpha=1, erase=True, save=True, step=10)
train_kwargs = dict(device=device, erase=True)
test_kwargs = dict(device=device, visualize=True, style='color', verbose=False, best_model='20', run=0, step=1, save_velocity=True)

data_generate_mesh(config, model, bc_pos, bc_dpos, **generate_kwargs)

# %% [markdown]
# The  GNN model (see src/PArticleGraph/models/Mesh_RPS.py) is optimized using the 'rock-paper-scissor' data.
#
# Since we ship the trained model with the repository, this step can be skipped if desired.
#
# %%
#| echo: true
#| output: false
if not os.path.exists(f'log/try_{config_file}'):
    data_train(config, config_file, **train_kwargs)

# %% [markdown]
# The model that has been trained in the previous step is used to generate the rollouts.
# %%
data_test(config, config_file, **test_kwargs)

# %% [markdown]
# Finally, we generate the figures that are shown in Figure 3.
# %%
#| echo: true
#| output: false
config_list, epoch_list = get_figures(figure_id, device=device)

# %%
#| fig-cap: "Initial configuration of the test training dataset. There are 4800 particles. The orange, blue, and green particles represent the three different particle types."
load_and_display('graphs_data/graphs_arbitrary_3/Fig/Fig_0_0.tif')

# %%
#| fig-cap: "Final configuration at frame 250"
load_and_display('graphs_data/graphs_arbitrary_3/Fig/Fig_0_250.tif')

# %%
#| fig-cap: "Learned latent vectors (x4800)"
load_and_display('log/try_arbitrary_3/results/embedding_arbitrary_3_20.tif')

# %%
#| fig-cap: "Learned interaction functions (x3)"
load_and_display('log/try_arbitrary_3/results/func_all_arbitrary_3_20.tif')


# %%
#| fig-cap: "GNN rollout inference at frame 250"
load_and_display('log/try_arbitrary_3/tmp_recons/Fig_arbitrary_3_249.tif')
