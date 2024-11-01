import glob
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import data
from tqdm import trange

from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.generators import choose_model
from ParticleGraph.models import get_index_particles
from ParticleGraph.plotting import load_and_display
from ParticleGraph.utils import CustomColorMap, fig_init, to_numpy, set_device


def init_particles(config, ratio, device):
    simulation_config = config.simulation
    n_particles = simulation_config.n_particles * ratio
    n_particle_types = simulation_config.n_particle_types
    dimension = simulation_config.dimension

    # Initialize position and velocity of the particles
    if simulation_config.boundary == 'periodic':
        pos = torch.rand(n_particles, dimension, device=device)
    else:
        pos = 0.5 * torch.randn(n_particles, dimension, device=device)
    velocity = simulation_config.dpos_init * torch.randn((n_particles, dimension), device=device)
    velocity_std = torch.std(velocity)
    velocity = torch.clamp(velocity, min=-velocity_std, max=velocity_std)

    # Initialize particle types (either all different, or given number of groups)
    if (simulation_config.params == 'continuous') | (simulation_config.non_discrete_level > 0):
        particle_type = torch.tensor(np.arange(n_particles), device=device)
    else:
        distinct_types = torch.arange(0, n_particle_types, dtype=torch.get_default_dtype(), device=device)
        particle_type = torch.repeat_interleave(distinct_types, int(n_particles / n_particle_types))
    particle_type = particle_type[:, None]

    # Initialize other particle properties
    features = torch.column_stack((torch.rand((n_particles, 1), device=device) , 0.1 * torch.randn((n_particles, 1), device=device)))
    age = torch.zeros((n_particles,1), device=device)
    particle_id = torch.arange(n_particles, device=device)[:, None]

    return pos, velocity, particle_type, features, age, particle_id


def data_generate_particle(config, visualize=True, run_vizualized=0, erase=False, step=5, ratio=1, device=None, bSave=True):
    simulation_config = config.simulation
    model_config = config.graph_model

    print(f'Generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
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

    # create GNN
    model, bc_pos, bc_dpos = choose_model(config=config, device=device)

    for run in range(config.training.n_runs):

        x_list = []
        y_list = []

        # initialize particle and graph states
        X1, V1, T1, H1, A1, N1 = init_particles(config=config, ratio=ratio, device=device)
        for it in trange(simulation_config.start_frame, n_frames + 1):

            x = torch.concatenate(
                (N1.clone().detach(), X1.clone().detach(), V1.clone().detach(), T1.clone().detach(),
                 H1.clone().detach(), A1.clone().detach()), 1)

            index_particles = get_index_particles(x, n_particle_types, dimension)  # can be different from frame to frame

            # compute connectivity rule
            distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
            adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index, field=[])

            # model prediction
            with torch.no_grad():
                y = model(dataset)

            # append list
            if (it >= 0) & bSave:
                x_list.append(x.clone().detach())
                y_list.append(y.clone().detach())

            # Particle update
            V1 = y
            X1 = bc_pos(X1 + V1 * delta_t)
            A1 = A1 + 1

            # output plots
            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):
                visualize_intermediate_state(cmap, dataset_name, index_particles, it, n_particle_types, run, x)

        if bSave:
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

    data_generate_particle(config, device=device, visualize=True, run_vizualized=0, erase=True, bSave=True, step=10)

    load_and_display('graphs_data/validation/Fig_0_0.tif')
    load_and_display('graphs_data/graphs_arbitrary_3/Fig/Fig_0_0.tif')

    load_and_display('graphs_data/validation/Fig_0_250.tif')
    load_and_display('graphs_data/graphs_arbitrary_3/Fig/Fig_0_250.tif')
