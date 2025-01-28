# %% [markdown]
# ---
# title: Attraction-Repulsion System with 64 Particle Types
# author: Cédric Allier, Michael Innerberger, Stephan Saalfeld
# categories:
#   - Particles
# execute:
#   echo: false
# image: "create_fig_3_6_files/figure-html/cell-10-output-1.png"
# ---

# %% [markdown]
# This script creates the fifth column of Figure 3 in the paper: we look at an attraction-repulsion system with 64
# particle types.

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
config_file = 'arbitrary_64'
figure_id = '3_6'
config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
device = set_device("auto")

# %% [markdown]
# The following model shows how the attraction-repulsion model is implemented in PyTorch Geometric.

# %%
#| echo: true
class AttractionRepulsionModel(pyg.nn.MessagePassing):
    """
    Compute the speed of particles as a function of their relative position according to an attraction-repulsion law.
    The latter is defined by four parameters p = (p1, p2, p3, p4) and a parameter sigma.

    See https://github.com/gpeyre/numerical-tours/blob/master/python/ml_10_particle_system.ipynb
    """

    def __init__(self, p, sigma, bc_dpos, dimension=2):
        super(AttractionRepulsionModel, self).__init__(aggr='mean')

        self.p = p
        self.sigma = sigma
        self.bc_dpos = bc_dpos
        self.dimension = dimension

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index

        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        particle_type = to_numpy(x[:, 1 + 2 * self.dimension])
        parameters = self.p[particle_type,:]
        d_pos = self.propagate(edge_index, pos=x[:, 1:self.dimension + 1], parameters=parameters)
        return d_pos


    def message(self, pos_i, pos_j, parameters_i):

        relative_position = self.bc_dpos(pos_j - pos_i)
        distance_squared = torch.sum(relative_position ** 2, dim=1)  # squared distance
        f = (parameters_i[:, 0] * torch.exp(-distance_squared ** parameters_i[:, 1] / (2 * self.sigma ** 2))
             - parameters_i[:, 2] * torch.exp(-distance_squared ** parameters_i[:, 3] / (2 * self.sigma ** 2)))
        velocity = f[:, None] * relative_position

        return velocity


def bc_pos(x):
    return torch.remainder(x, 1.0)


def bc_dpos(x):
    return torch.remainder(x - 0.5, 1.0) - 0.5

# %% [markdown]
# Subsequently, the data is generated, and the model is trained and tested.
# Since we ship the trained model with the repository, this step can be skipped if desired.
# The training model can be found at [Interaction_particle.py](https://github.com/saalfeldlab/decomp-gnn/blob/main/src/ParticleGraph/models/Interaction_Particle.py).

# %%
#| echo: true
#| output: false
p = torch.squeeze(torch.tensor(config.simulation.params))
sigma = config.simulation.sigma
model = AttractionRepulsionModel(
    p=p,
    sigma=sigma,
    bc_dpos=bc_dpos,
    dimension=config.simulation.dimension
)

generate_kwargs = dict(device=device, visualize=True, run_vizualized=0, style='color', alpha=1, erase=True, save=True, step=10)
train_kwargs = dict(device=device, erase=True)
test_kwargs = dict(device=device, visualize=True, style='color', verbose=False, best_model='20', run=0, step=1, save_velocity=True)

data_generate_particles(config, model, bc_pos, bc_dpos, **generate_kwargs)
if not os.path.exists(f'log/try_{config_file}'):
    data_train(config, config_file, **train_kwargs)
data_test(config, config_file, **test_kwargs)

# %% [markdown]
# Finally, we generate the figures that are shown in the first column of Figure 3. The model that has been trained in the
# previous step is used to generate the rollouts.

# %%
#| echo: true
#| output: false
config_list, epoch_list = get_figures(figure_id, device=device)

# %%
#| fig-cap: "Initial configuration of the test training dataset. There are 4800 particles. Colors indicate the particle types."
load_and_display('graphs_data/graphs_arbitrary_64/Fig/Fig_0_0.tif')

# %%
#| fig-cap: "Final configuration at frame 1000"
load_and_display('graphs_data/graphs_arbitrary_64/Fig/Fig_0_1000.tif')

# %%
#| fig-cap: "Learned latent vectors (x4800)"
load_and_display('log/try_arbitrary_64/results/embedding_arbitrary_64_20.tif')

# %%
#| fig-cap: "Learned interaction functions (x64)"
load_and_display('log/try_arbitrary_64/results/func_all_arbitrary_64_20.tif')

# %%
#| fig-cap: "Initial random configuration for rollout"
load_and_display('log/try_arbitrary_64/tmp_recons/Fig_arbitrary_64_0.tif')

# %%
#| fig-cap: "Final configuration in rollout at frame 1000"
load_and_display('log/try_arbitrary_64/tmp_recons/Fig_arbitrary_64_999.tif')
