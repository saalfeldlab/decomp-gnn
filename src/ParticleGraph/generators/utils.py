# matplotlib.use("Qt5Agg")
from time import sleep

import numpy as np
import torch
from scipy.spatial import Delaunay
from tifffile import imread
from torch_geometric.utils import get_mesh_laplacian
from tqdm import trange

from ParticleGraph.data_loaders import load_solar_system
from ParticleGraph.generators import PDE_A, PDE_B, PDE_E, PDE_G, PDE_N, PDE_Z, RD_RPS, PDE_Laplacian
from ParticleGraph.utils import choose_boundary_values
from ParticleGraph.utils import to_numpy


def generate_from_data(config, device, visualize=True, folder=None, step=None):

    data_folder_name = config.data_folder_name

    match data_folder_name:
        case 'graphs_data/solar_system':
            load_solar_system(config, device, visualize, folder, step)
        case _:
            raise ValueError(f'Unknown data folder name {data_folder_name}')


def choose_model(config=[], W=[], phi=[], device=[]):
    particle_model_name = config.graph_model.particle_model_name
    model_signal_name = config.graph_model.signal_model_name
    aggr_type = config.graph_model.aggr_type
    n_particles = config.simulation.n_particles
    n_node_types = config.simulation.n_node_types
    n_nodes = config.simulation.n_nodes
    n_particle_types = config.simulation.n_particle_types
    bc_pos, bc_dpos = choose_boundary_values(config.simulation.boundary)
    dimension = config.simulation.dimension

    params = config.simulation.params

    # create GNN depending in type specified in config file
    match particle_model_name:
        case 'PDE_A' | 'PDE_ParticleField_A' | 'PDE_Cell_A' :
            p = torch.ones(n_particle_types, 4, device=device) + torch.rand(n_particle_types, 4, device=device)
            if config.simulation.non_discrete_level>0:
                pp=[]
                n_particle_types = len(params)
                for n in range(n_particle_types):
                    p[n] = torch.tensor(params[n])
                for n in range(n_particle_types):
                    if n==0:
                        pp=p[n].repeat(n_particles//n_particle_types,1)
                    else:
                        pp=torch.cat((pp,p[n].repeat(n_particles//n_particle_types,1)),0)
                p=pp.clone().detach()
                p=p+torch.randn(n_particles,4,device=device) * config.simulation.non_discrete_level
            elif params[0] != [-1]:
                for n in range(n_particle_types):
                    p[n] = torch.tensor(params[n])
            else:
                print(p)
            sigma = config.simulation.sigma
            p = p if n_particle_types == 1 else torch.squeeze(p)
            model = PDE_A(aggr_type=aggr_type, p=torch.squeeze(p), sigma=sigma, bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_B' | 'PDE_ParticleField_B' | 'PDE_Cell_B' | 'PDE_Cell_B_area':
            p = torch.rand(n_particle_types, 3, device=device) * 100  # comprised between 10 and 50
            if params[0] != [-1]:
                for n in range(n_particle_types):
                    p[n] = torch.tensor(params[n])
            else:
                print(p)
            model = PDE_B(aggr_type=aggr_type, p=torch.squeeze(p), bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_G':
            if params[0] == [-1]:
                p = np.linspace(0.5, 5, n_particle_types)
                p = torch.tensor(p, device=device)
            if len(params) > 1:
                for n in range(n_particle_types):
                    p[n] = torch.tensor(params[n])
            model = PDE_G(aggr_type=aggr_type, p=torch.squeeze(p), clamp=config.training.clamp,
                          pred_limit=config.training.pred_limit, bc_dpos=bc_dpos)
        case 'PDE_E':
            p = initialize_random_values(n_particle_types, device)
            if len(params) > 0:
                for n in range(n_particle_types):
                    p[n] = torch.tensor(params[n])
            model = PDE_E(aggr_type=aggr_type, p=torch.squeeze(p),
                          clamp=config.training.clamp, pred_limit=config.training.pred_limit,
                          prediction=config.graph_model.prediction, bc_dpos=bc_dpos)
        case _:
            model = PDE_Z(device=device)

    match model_signal_name:
        case 'PDE_N':
            p = torch.rand(n_particle_types, 2, device=device) * 100  # comprised between 10 and 50
            if params[0] != [-1]:
                for n in range(n_particle_types):
                    p[n] = torch.tensor(params[n])
            model = PDE_N(aggr_type=aggr_type, p=torch.squeeze(p), bc_dpos=bc_dpos)



    return model, bc_pos, bc_dpos


def choose_mesh_model(config, X1_mesh, device):
    mesh_model_name = config.graph_model.mesh_model_name
    n_node_types = config.simulation.n_node_types
    aggr_type = config.graph_model.mesh_aggr_type
    _, bc_dpos = choose_boundary_values(config.simulation.boundary)

    if mesh_model_name =='':
        mesh_model = []
    else:
        # c = initialize_random_values(n_node_types, device)
        # if not('pics' in config.simulation.node_coeff_map):
        #     for n in range(n_node_types):
        #         c[n] = torch.tensor(config.simulation.diffusion_coefficients[n])

        beta = config.simulation.beta

        match mesh_model_name:
            case 'RD_RPS_Mesh':
                mesh_model = RD_RPS(aggr_type=aggr_type, bc_dpos=bc_dpos)
            case 'RD_RPS_Mesh_bis':
                mesh_model = RD_RPS(aggr_type=aggr_type, bc_dpos=bc_dpos)
            case 'DiffMesh' | 'WaveMesh':
                mesh_model = PDE_Laplacian(aggr_type=aggr_type, beta=beta, bc_dpos=bc_dpos)
            case _:
                mesh_model = PDE_Z(device=device)


        i0 = imread(f'graphs_data/{config.simulation.node_coeff_map}')
        i0 = np.flipud(i0)
        values = i0[(to_numpy(X1_mesh[:, 1]) * 255).astype(int), (to_numpy(X1_mesh[:, 0]) * 255).astype(int)]
        values = np.reshape(values,len(X1_mesh))
        mesh_model.coeff = torch.tensor(values, device=device, dtype=torch.float32)[:, None]

    return mesh_model


# TODO: this seems to be used to provide default values in case no parameters are given?
def initialize_random_values(n, device):
    return torch.ones(n, 1, device=device) + torch.rand(n, 1, device=device)


def init_particles(config=[], scenario='none', ratio=1, device=[]):
    simulation_config = config.simulation
    n_frames = config.simulation.n_frames
    n_particles = simulation_config.n_particles * ratio
    n_particle_types = simulation_config.n_particle_types
    dimension = simulation_config.dimension

    dpos_init = simulation_config.dpos_init

    if (simulation_config.boundary == 'periodic'): # | (simulation_config.dimension == 3):
        pos = torch.rand(n_particles, dimension, device=device)
    else:
        pos = torch.randn(n_particles, dimension, device=device) * 0.5
    dpos = dpos_init * torch.randn((n_particles, dimension), device=device)
    dpos = torch.clamp(dpos, min=-torch.std(dpos), max=+torch.std(dpos))
    type = torch.zeros(int(n_particles / n_particle_types), device=device)
    for n in range(1, n_particle_types):
        type = torch.cat((type, n * torch.ones(int(n_particles / n_particle_types), device=device)), 0)
    if (simulation_config.params == 'continuous') | (config.simulation.non_discrete_level > 0):  # TODO: params is a list[list[float]]; this can never happen?
        type = torch.tensor(np.arange(n_particles), device=device)

    features = torch.cat((torch.rand((n_particles, 1), device=device) , 0.1 * torch.randn((n_particles, 1), device=device)), 1)

    type = type[:, None]
    particle_id = torch.arange(n_particles, device=device)
    particle_id = particle_id[:, None]
    age = torch.zeros((n_particles,1), device=device)

    if 'pattern' in scenario:
        i0 = imread(f'graphs_data/pattern_0.tif')
        type = np.round(i0[(to_numpy(pos[:, 0]) * 255).astype(int), (to_numpy(pos[:, 1]) * 255).astype(int)] / 255 * n_particle_types-1).astype(int)
        type = torch.tensor(type, device=device)
        type = type[:, None]
    if 'uniform' in scenario :

        type = torch.ones(n_particles, device=device) * int(scenario.split()[-1])
        type =  type[:, None]
    if 'stripes' in scenario:
        l = n_particles//n_particle_types
        for n in range(n_particle_types):
            index = np.arange(n*l, (n+1)*l)
            pos[index, 1:2] = torch.rand(l, 1, device=device) * (1/n_particle_types) + n/n_particle_types


    return pos, dpos, type, features, age, particle_id


def get_index(n_particles, n_particle_types):
    index_particles = []
    for n in range(n_particle_types):
        index_particles.append(
            np.arange((n_particles // n_particle_types) * n, (n_particles // n_particle_types) * (n + 1)))
    return index_particles


def get_time_series(x_list, cell_id, feature):

    match feature:
        case 'velocity_x':
            feature = 3
        case 'velocity_y':
            feature = 4
        case 'type' | 'state':
            feature = 5
        case 'age':
            feature = 8
        case 'mass':
            feature = 10

        case _:  # default
            feature = 0

    time_series = []
    for it in range(len(x_list)):
        x = x_list[it].clone().detach()
        pos_cell = torch.argwhere(x[:, 0] == cell_id)
        if len(pos_cell) > 0:
            time_series.append(x[pos_cell, feature].squeeze())
        else:
            time_series.append(torch.tensor([0.0]))

    return to_numpy(torch.stack(time_series))


def init_mesh(config, device):
    simulation_config = config.simulation
    n_nodes = simulation_config.n_nodes
    n_particles = simulation_config.n_particles
    node_value_map = simulation_config.node_value_map
    node_coeff_map = simulation_config.node_coeff_map

    n_nodes_per_axis = int(np.sqrt(n_nodes))
    xs = torch.linspace(1 / (2 * n_nodes_per_axis), 1 - 1 / (2 * n_nodes_per_axis), steps=n_nodes_per_axis)
    ys = torch.linspace(1 / (2 * n_nodes_per_axis), 1 - 1 / (2 * n_nodes_per_axis), steps=n_nodes_per_axis)
    x_mesh, y_mesh = torch.meshgrid(xs, ys, indexing='xy')
    x_mesh = torch.reshape(x_mesh, (n_nodes_per_axis ** 2, 1))
    y_mesh = torch.reshape(y_mesh, (n_nodes_per_axis ** 2, 1))
    mesh_size = 1 / n_nodes_per_axis
    pos_mesh = torch.zeros((n_nodes, 2), device=device)
    pos_mesh[0:n_nodes, 0:1] = x_mesh[0:n_nodes]
    pos_mesh[0:n_nodes, 1:2] = y_mesh[0:n_nodes]

    i0 = imread(f'graphs_data/{node_value_map}')
    if 'video' in simulation_config.node_value_map:
        i0 = imread(f'graphs_data/pattern_Null.tif')
    else:
        i0 = imread(f'graphs_data/{node_value_map}')
        i0 = np.flipud(i0)
    values = i0[(to_numpy(pos_mesh[:, 1]) * 255).astype(int), (to_numpy(pos_mesh[:, 0]) * 255).astype(int)]

    mask_mesh = (x_mesh > torch.min(x_mesh) + 0.02) & (x_mesh < torch.max(x_mesh) - 0.02) & (y_mesh > torch.min(y_mesh) + 0.02) & (y_mesh < torch.max(y_mesh) - 0.02)

    pos_mesh = pos_mesh + torch.randn(n_nodes, 2, device=device) * mesh_size / 8

    match config.graph_model.mesh_model_name:
        case 'RD_Gray_Scott_Mesh':
            features_mesh = torch.zeros((n_nodes, 2), device=device)
            features_mesh[:, 0] -= 0.5 * torch.tensor(values / 255, device=device)
            features_mesh[:, 1] = 0.25 * torch.tensor(values / 255, device=device)
        case 'RD_FitzHugh_Nagumo_Mesh':
            features_mesh = torch.zeros((n_nodes, 2), device=device) + torch.rand((n_nodes, 2), device=device) * 0.1
        case 'RD_RPS_Mesh' | 'RD_RPS_Mesh_bis':
            features_mesh = torch.rand((n_nodes, 3), device=device)
            s = torch.sum(features_mesh, dim=1)
            for k in range(3):
                features_mesh[:, k] = features_mesh[:, k] / s
        case '' | 'DiffMesh' | 'WaveMesh' | 'Particle_Mesh_A' | 'Particle_Mesh_B':
            features_mesh = torch.zeros((n_nodes, 2), device=device)
            features_mesh[:, 0] = torch.tensor(values / 255 * 5000, device=device)
        case 'PDE_O_Mesh':
            features_mesh = torch.zeros((n_particles, 5), device=device)
            features_mesh[0:n_particles, 0:1] = x_mesh[0:n_particles]
            features_mesh[0:n_particles, 1:2] = y_mesh[0:n_particles]
            features_mesh[0:n_particles, 2:3] = torch.randn(n_particles, 1, device=device) * 2 * np.pi  # theta
            features_mesh[0:n_particles, 3:4] = torch.ones(n_particles, 1, device=device) * np.pi / 200  # d_theta
            features_mesh[0:n_particles, 4:5] = features_mesh[0:n_particles, 3:4]  # d_theta0
            pos_mesh[:, 0] = features_mesh[:, 0] + (3 / 8) * mesh_size * torch.cos(features_mesh[:, 2])
            pos_mesh[:, 1] = features_mesh[:, 1] + (3 / 8) * mesh_size * torch.sin(features_mesh[:, 2])

    # i0 = imread(f'graphs_data/{node_coeff_map}')
    # values = i0[(to_numpy(x_mesh[:, 0]) * 255).astype(int), (to_numpy(y_mesh[:, 0]) * 255).astype(int)]
    # type_mesh = torch.tensor(values, device=device)
    # type_mesh = type_mesh[:, None]

    # i0 = imread(f'graphs_data/{node_coeff_map}')
    # values = i0[(to_numpy(x_mesh[:, 0]) * 255).astype(int), (to_numpy(y_mesh[:, 0]) * 255).astype(int)]
    # if np.max(values) > 0:
    #     values = np.round(values / np.max(values) * (simulation_config.n_node_types-1))
    # type_mesh = torch.tensor(values, device=device)
    # type_mesh = type_mesh[:, None]

    type_mesh = torch.zeros((n_nodes, 1), device=device)

    node_id_mesh = torch.arange(n_nodes, device=device)
    node_id_mesh = node_id_mesh[:, None]
    dpos_mesh = torch.zeros((n_nodes, 2), device=device)

    x_mesh = torch.concatenate((node_id_mesh.clone().detach(), pos_mesh.clone().detach(), dpos_mesh.clone().detach(),
                                type_mesh.clone().detach(), features_mesh.clone().detach()), 1)

    pos = to_numpy(x_mesh[:, 1:3])
    tri = Delaunay(pos, qhull_options='QJ')
    face = torch.from_numpy(tri.simplices)
    face_longest_edge = np.zeros((face.shape[0], 1))

    print('Removal of skinny faces ...')
    sleep(0.5)
    for k in trange(face.shape[0]):
        # compute edge distances
        x1 = pos[face[k, 0], :]
        x2 = pos[face[k, 1], :]
        x3 = pos[face[k, 2], :]
        a = np.sqrt(np.sum((x1 - x2) ** 2))
        b = np.sqrt(np.sum((x2 - x3) ** 2))
        c = np.sqrt(np.sum((x3 - x1) ** 2))
        A = np.max([a, b]) / np.min([a, b])
        B = np.max([a, c]) / np.min([a, c])
        C = np.max([c, b]) / np.min([c, b])
        face_longest_edge[k] = np.max([A, B, C])

    face_kept = np.argwhere(face_longest_edge < 5)
    face_kept = face_kept[:, 0]
    face = face[face_kept, :]
    face = face.t().contiguous()
    face = face.to(device, torch.long)

    pos_3d = torch.cat((x_mesh[:, 1:3], torch.ones((x_mesh.shape[0], 1), device=device)), dim=1)
    edge_index_mesh, edge_weight_mesh = get_mesh_laplacian(pos=pos_3d, face=face, normalization="None")
    edge_weight_mesh = edge_weight_mesh.to(dtype=torch.float32)
    mesh_data = {'mesh_pos': pos_3d, 'face': face, 'edge_index': edge_index_mesh, 'edge_weight': edge_weight_mesh,
                 'mask': mask_mesh, 'size': mesh_size}

    if (config.graph_model.particle_model_name == 'PDE_ParticleField_A')  | (config.graph_model.particle_model_name == 'PDE_ParticleField_B'):
        type_mesh = 0 * type_mesh

    # if config.graph_model.particle_model_name == 'PDE_ParticleField_B':
    #
    #     a1 = 1E-2  # diffusion coefficient
    #     a2 = 8E-5  # positive rate coefficient
    #     a3 = 6.65E-5  # negative rate coefficient
    #
    #     i0 = imread(f'graphs_data/{config.simulation.node_diffusion_map}')
    #     index = np.round(
    #         i0[(to_numpy(x_mesh[:, 1]) * 255).astype(int), (to_numpy(x_mesh[:, 2]) * 255).astype(int)]).astype(int)
    #     coeff_diff = a1 * np.array(config.simulation.diffusion_coefficients)[index]
    #     model.coeff_diff = torch.tensor(coeff_diff, device=device)
    #     i0 = imread(f'graphs_data/{config.simulation.node_proliferation_map}')
    #     index = np.round(
    #         i0[(to_numpy(x_mesh[:, 0]) * 255).astype(int), (to_numpy(x_mesh[:, 1]) * 255).astype(int)]).astype(int)
    #     pos_rate = a2 * np.array(config.simulation.pos_rate)[index]
    #     model.pos_rate = torch.tensor(pos_rate, device=device)
    #     model.neg_rate = - torch.ones_like(model.pos_rate) * a3 * torch.tensor(config.simulation.pos_rate[0], device=device)
    #
    #     type_mesh = -1.0 + type_mesh * -1.0

    a_mesh = torch.zeros_like(type_mesh)
    type_mesh = type_mesh.to(dtype=torch.float32)


    return pos_mesh, dpos_mesh, type_mesh, features_mesh, a_mesh, node_id_mesh, mesh_data

