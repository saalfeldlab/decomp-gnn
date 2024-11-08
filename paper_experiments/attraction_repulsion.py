import glob
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
import torch_geometric as pyg
from torch_geometric.data import data
import torch_geometric.utils as pyg_utils
from tqdm import trange

from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.plotting import load_and_display
from ParticleGraph.utils import CustomColorMap, fig_init, to_numpy, set_device


class AttractionRepulsionModel(pyg.nn.MessagePassing):
    """
    Compute the speed of particles as a function of their relative position according to an attraction-repulsion law.
    The latter is defined by four parameters p = (p1, p2, p3, p4) and a parameter sigma.

    See https://github.com/gpeyre/numerical-tours/blob/master/python/ml_10_particle_system.ipynb
    """

    def __init__(self, p, sigma, bc_dpos):
        super(AttractionRepulsionModel, self).__init__(aggr='mean')

        self.p = p
        self.sigma = sigma
        self.bc_dpos = bc_dpos

    def forward(self, data: data):
        x, edge_index = data.x, data.edge_index

        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        particle_type = torch.squeeze(data.particle_type).int()
        parameters = self.p[particle_type,:]
        d_pos = self.propagate(edge_index, pos=data.pos, parameters=parameters)
        return d_pos

    # noinspection PyMethodOverriding
    def message(self, pos_i, pos_j, parameters_i):

        relative_position = self.bc_dpos(pos_j - pos_i)
        distance_squared = torch.sum(relative_position ** 2, dim=1)  # squared distance
        f = (parameters_i[:, 0] * torch.exp(-distance_squared ** parameters_i[:, 1] / (2 * self.sigma ** 2))
             - parameters_i[:, 2] * torch.exp(-distance_squared ** parameters_i[:, 3] / (2 * self.sigma ** 2)))
        velocity = f[:, None] * relative_position

        return velocity


def bc_pos(x):
    return torch.remainder(x, 1.0)  # in [0, 1)


def bc_dpos(x):
    return torch.remainder(x - 0.5, 1.0) - 0.5  # in [-0.5, 0.5)


def get_index_particles(particle_type, n_particle_types):
    index_particles = []
    for n in range(n_particle_types):
        index = torch.argwhere(n == particle_type)
        index_particles.append(index.squeeze())
    return index_particles


def init_particles(config, ratio):
    simulation_config = config.simulation
    n_particles = simulation_config.n_particles * ratio
    n_particle_types = simulation_config.n_particle_types
    dimension = simulation_config.dimension

    # Initialize position and velocity of the particles
    if simulation_config.boundary == 'periodic':
        pos = torch.rand(n_particles, dimension)
    else:
        pos = 0.5 * torch.randn(n_particles, dimension)
    velocity = simulation_config.dpos_init * torch.randn((n_particles, dimension))
    velocity_std = torch.std(velocity)
    velocity = torch.clamp(velocity, min=-velocity_std, max=velocity_std)

    # Initialize particle types (either all different, or given number of groups)
    if (simulation_config.params == 'continuous') | (simulation_config.non_discrete_level > 0):
        particle_type = torch.tensor(np.arange(n_particles))
    else:
        distinct_types = torch.arange(0, n_particle_types, dtype=torch.get_default_dtype())
        particle_type = torch.repeat_interleave(distinct_types, int(n_particles / n_particle_types))
    particle_type = particle_type[:, None]

    # Initialize other particle properties
    features = torch.column_stack((torch.rand((n_particles, 1)) , 0.1 * torch.randn((n_particles, 1))))
    age = torch.zeros((n_particles,1))
    particle_id = torch.arange(n_particles)[:, None]

    return pos, velocity, particle_type, features, age, particle_id


def data_generate_particle(config, model, visualize=True, run_visualized=0, erase=False, step=5, ratio=1, save=True):
    simulation_config = config.simulation
    model_config = config.graph_model

    print(f'Generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    delta_t = simulation_config.delta_t
    n_frames = simulation_config.n_frames
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset

    folder = f'./graphs_data/graphs_{dataset_name}/'
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (f[-3:] != 'Fig') & (f[-14:] != 'generated_data') & (f != 'p.pt') & (f != 'cycle_length.pt') & (f != 'model_config.json') & (f != 'generation_code.py'):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/graphs_{dataset_name}/Fig/', exist_ok=True)
    files = glob.glob(f'./graphs_data/graphs_{dataset_name}/Fig/*')
    for f in files:
        os.remove(f)

    for run in range(config.training.n_runs):

        x_list = []
        y_list = []

        # initialize particle and graph states
        pos, velocity, particle_type, features, age, particle_id = init_particles(config=config, ratio=ratio)

        for it in trange(simulation_config.start_frame, n_frames + 1):

            # compute connectivity rule
            dataset = data.Data(pos=pos.detach().clone(), particle_type=particle_type.detach().clone())
            distance = torch.sum(bc_dpos(dataset.pos[:, None, :] - dataset.pos[None, :, :]) ** 2, dim=2)
            adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            dataset.edge_index = edge_index

            # model prediction
            with torch.no_grad():
                y = model(dataset)

            # append list
            x = torch.concatenate([t.detach().clone() for t in (particle_id, pos, velocity, particle_type, features, age)], 1)
            if (it >= 0) & save:
                x_list.append(x)
                y_list.append(y.detach().clone())

            # Particle update
            velocity = y
            pos = bc_pos(pos + velocity * delta_t)
            age = age + 1

            # output plots
            if visualize & (run == run_visualized) & (it % step == 0) & (it >= 0):
                index_particles = get_index_particles(dataset.particle_type, n_particle_types)
                visualize_intermediate_state(cmap, dataset_name, index_particles, it, n_particle_types, run, x)

        if save:
            torch.save(x_list, f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt')
            torch.save(y_list, f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt')
            torch.save(model.p, f'graphs_data/graphs_{dataset_name}/model_p.pt')


def visualize_intermediate_state(cmap, dataset_name, index_particles, it, n_particle_types, run, x):

        fig, ax = fig_init(formatx="%.1f", formaty="%.1f")
        s_p = 100
        for n in range(n_particle_types):
            plt.scatter(to_numpy(x[index_particles[n], 2]), to_numpy(x[index_particles[n], 1]),
                        s=s_p, color=cmap.color(n))
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('x', fontsize=48)
        plt.ylabel('y', fontsize=48)
        plt.xticks(fontsize=48.0)
        plt.yticks(fontsize=48.0)
        ax.tick_params(axis='both', which='major', pad=15)
        plt.text(0, 1.1, f'frame {it}', ha='left', va='top', transform=ax.transAxes, fontsize=48)
        plt.tight_layout()
        plt.savefig(f"graphs_data/graphs_{dataset_name}/Fig/Fig_{run}_{it}.tif", dpi=80)
        plt.close()


if __name__ == '__main__':

    torch.manual_seed(0)

    config_file = 'arbitrary_3'
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    device = set_device("auto")

    with torch.device(device):
        model = AttractionRepulsionModel(
            p=torch.tensor(config.simulation.params),
            sigma=config.simulation.sigma,
            bc_dpos=bc_dpos
        )
        data_generate_particle(config, model, visualize=True, run_visualized=0, erase=True, save=True, step=10)

    load_and_display('graphs_data/validation/Fig_0_0.tif')
    load_and_display('graphs_data/graphs_arbitrary_3/Fig/Fig_0_0.tif')

    load_and_display('graphs_data/validation/Fig_0_250.tif')
    load_and_display('graphs_data/graphs_arbitrary_3/Fig/Fig_0_250.tif')
