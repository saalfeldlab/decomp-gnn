import sys
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from torch_geometric.nn import MessagePassing
import torch_geometric.utils as pyg_utils
from matplotlib import rc
import matplotlib as mpl
from io import StringIO
from scipy.stats import pearsonr
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.ndimage import median_filter

from ParticleGraph.fitting_models import *
from ParticleGraph.utils import set_size
from ParticleGraph.sparsify import *
from ParticleGraph.models.utils import *
from ParticleGraph.models.MLP import *
from ParticleGraph.utils import to_numpy, CustomColorMap

os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/x86_64-linux'

# matplotlib.use("Qt5Agg")

class Interaction_Particle_extract(MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, config, device, aggr_type=None, bc_dpos=None):

        super(Interaction_Particle_extract, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        config.simulation = config.simulation
        config.graph_model = config.graph_model
        config.training = config.training

        self.device = device
        self.input_size = config.graph_model.input_size
        self.output_size = config.graph_model.output_size
        self.hidden_dim = config.graph_model.hidden_dim
        self.n_layers = config.graph_model.n_mp_layers
        self.n_particles = config.simulation.n_particles
        self.max_radius = config.simulation.max_radius
        self.data_augmentation = config.training.data_augmentation
        self.noise_level = config.training.noise_level
        self.embedding_dim = config.graph_model.embedding_dim
        self.n_dataset = config.training.n_runs
        self.prediction = config.graph_model.prediction
        self.update_type = config.graph_model.update_type
        self.n_layers_update = config.graph_model.n_layers_update
        self.hidden_dim_update = config.graph_model.hidden_dim_update
        self.sigma = config.simulation.sigma
        self.model = config.graph_model.particle_model_name
        self.bc_dpos = bc_dpos
        self.n_ghosts = int(config.training.n_ghosts)
        self.n_particles_max = config.simulation.n_particles_max

        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                            hidden_size=self.hidden_dim, device=self.device)

        if config.simulation.has_cell_division:
            self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, self.n_particles_max, 2)), device=self.device,
                             requires_grad=True, dtype=torch.float32))
        else:
            self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, int(self.n_particles) + self.n_ghosts, self.embedding_dim)),
                             device=self.device,
                             requires_grad=True, dtype=torch.float32))

        if self.update_type != 'none':
            self.lin_update = MLP(input_size=self.output_size + self.embedding_dim + 2, output_size=self.output_size,
                                  nlayers=self.n_layers_update, hidden_size=self.hidden_dim_update, device=self.device)

    def forward(self, data=[], data_id=[], training=[], vnorm=[], phi=[], has_field=False):

        self.data_id = data_id
        self.vnorm = vnorm
        self.cos_phi = torch.cos(phi)
        self.sin_phi = torch.sin(phi)
        self.training = training
        self.has_field = has_field

        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        pos = x[:, 1:3]
        d_pos = x[:, 3:5]
        particle_id = x[:, 0:1]
        if has_field:
            field = x[:, 6:7]
        else:
            field = torch.ones_like(x[:, 6:7])

        pred = self.propagate(edge_index, pos=pos, d_pos=d_pos, particle_id=particle_id, field=field)

        return pred, self.in_features, self.lin_edge_out

    def message(self, pos_i, pos_j, d_pos_i, d_pos_j, particle_id_i, particle_id_j, field_j):
        # squared distance
        r = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1)) / self.max_radius
        delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius
        dpos_x_i = d_pos_i[:, 0] / self.vnorm
        dpos_y_i = d_pos_i[:, 1] / self.vnorm
        dpos_x_j = d_pos_j[:, 0] / self.vnorm
        dpos_y_j = d_pos_j[:, 1] / self.vnorm

        if self.data_augmentation & (self.training == True):
            new_delta_pos_x = self.cos_phi * delta_pos[:, 0] + self.sin_phi * delta_pos[:, 1]
            new_delta_pos_y = -self.sin_phi * delta_pos[:, 0] + self.cos_phi * delta_pos[:, 1]
            delta_pos[:, 0] = new_delta_pos_x
            delta_pos[:, 1] = new_delta_pos_y
            new_dpos_x_i = self.cos_phi * dpos_x_i + self.sin_phi * dpos_y_i
            new_dpos_y_i = -self.sin_phi * dpos_x_i + self.cos_phi * dpos_y_i
            dpos_x_i = new_dpos_x_i
            dpos_y_i = new_dpos_y_i
            new_dpos_x_j = self.cos_phi * dpos_x_j + self.sin_phi * dpos_y_j
            new_dpos_y_j = -self.sin_phi * dpos_x_j + self.cos_phi * dpos_y_j
            dpos_x_j = new_dpos_x_j
            dpos_y_j = new_dpos_y_j

        embedding_i = self.a[self.data_id, to_numpy(particle_id_i), :].squeeze()
        embedding_j = self.a[self.data_id, to_numpy(particle_id_j), :].squeeze()

        match self.model:
            case 'PDE_A':
                in_features = torch.cat((delta_pos, r[:, None], embedding_i), dim=-1)
            case 'PDE_B' | 'PDE_B_bis' | 'PDE_Cell_B':
                in_features = torch.cat((delta_pos, r[:, None], dpos_x_i[:, None], dpos_y_i[:, None], dpos_x_j[:, None],
                                         dpos_y_j[:, None], embedding_i), dim=-1)
            case 'PDE_G':
                in_features = torch.cat((delta_pos, r[:, None], dpos_x_i[:, None], dpos_y_i[:, None],
                                         dpos_x_j[:, None], dpos_y_j[:, None], embedding_j), dim=-1)
            case 'PDE_GS':
                in_features = torch.cat((r[:, None], embedding_j), dim=-1)
            case 'PDE_E':
                in_features = torch.cat(
                    (delta_pos, r[:, None], embedding_i, embedding_j), dim=-1)

        out = self.lin_edge(in_features) * field_j

        self.in_features = in_features
        self.lin_edge_out = out

        return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p):

        if (len(p) == 3):  # PDE_B
            cohesion = p[0] * 0.5E-5 * r
            separation = -p[2] * 1E-8 / r
            return (cohesion + separation) * p[1] / 500  #
        else:  # PDE_A
            return r * (p[0] * torch.exp(-r ** (2 * p[1]) / (2 * self.sigma ** 2)) - p[2] * torch.exp(
                -r ** (2 * p[3]) / (2 * self.sigma ** 2)))


class model_qiqj(nn.Module):

    def __init__(self, size=None, device=None):

        super(model_qiqj, self).__init__()

        self.device = device
        self.size = size

        self.qiqj = nn.Parameter(torch.randn((int(self.size), 1), device=self.device,requires_grad=True, dtype=torch.float32))


    def forward(self):

        x = []
        for l in range(self.size):
            for m in range(l,self.size,1):
                x.append(self.qiqj[l] * self.qiqj[m])

        return torch.stack(x)


class PDE_B_extract(MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=None, p=None, bc_dpos=None):
        super(PDE_B_extract, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.bc_dpos = bc_dpos

        self.a1 = 0.5E-5
        self.a2 = 5E-4
        self.a3 = 1E-8

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        acc = self.propagate(edge_index, x=x)

        sum = self.cohesion + self.alignment + self.separation

        return acc, sum, self.cohesion, self.alignment, self.separation, self.diffx, self.diffv, self.r, self.type

    def message(self, x_i, x_j):
        r = torch.sum(self.bc_dpos(x_j[:, 1:3] - x_i[:, 1:3]) ** 2, dim=1)  # distance squared

        pp = self.p[to_numpy(x_i[:, 5]), :]

        cohesion = pp[:, 0:1].repeat(1, 2) * self.a1 * self.bc_dpos(x_j[:, 1:3] - x_i[:, 1:3])
        alignment = pp[:, 1:2].repeat(1, 2) * self.a2 * self.bc_dpos(x_j[:, 3:5] - x_i[:, 3:5])
        separation = pp[:, 2:3].repeat(1, 2) * self.a3 * self.bc_dpos(x_i[:, 1:3] - x_j[:, 1:3]) / (
            r[:, None].repeat(1, 2))

        self.cohesion = cohesion
        self.alignment = alignment
        self.separation = separation

        self.r = r
        self.diffx = self.bc_dpos(x_j[:, 1:3] - x_i[:, 1:3])
        self.diffv = self.bc_dpos(x_j[:, 3:5] - x_i[:, 3:5])
        self.type = x_i[:, 5]

        return (separation + alignment + cohesion)

    def psi(self, r, p):
        cohesion = p[0] * self.a1 * r
        separation = -p[2] * self.a3 / r
        return (cohesion + separation)


class Mesh_RPS_extract(MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    def __init__(self, aggr_type=None, config=None, device=None, bc_dpos=None):
        super(Mesh_RPS_extract, self).__init__(aggr=aggr_type)

        config.simulation = config.simulation
        config.graph_model = config.graph_model

        self.device = device
        self.input_size = config.graph_model.input_size
        self.output_size = config.graph_model.output_size
        self.hidden_size = config.graph_model.hidden_dim
        self.nlayers = config.graph_model.n_mp_layers
        self.embedding_dim = config.graph_model.embedding_dim
        self.nparticles = config.simulation.n_particles
        self.ndataset = config.training.n_runs
        self.bc_dpos = bc_dpos

        self.lin_phi = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.nlayers,
                           hidden_size=self.hidden_size, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((int(self.ndataset), int(self.nparticles), self.embedding_dim)), device=self.device,
                         requires_grad=True, dtype=torch.float32))

    def forward(self, data, data_id):
        self.data_id = data_id
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        uvw = data.x[:, 6:9]

        laplacian_uvw = self.propagate(edge_index, uvw=uvw, discrete_laplacian=edge_attr)

        particle_id = to_numpy(x[:, 0])
        embedding = self.a[self.data_id, particle_id, :]

        input_phi = torch.cat((laplacian_uvw, uvw, embedding), dim=-1)

        pred = self.lin_phi(input_phi)

        return pred, input_phi, embedding

    def message(self, uvw_j, discrete_laplacian):
        return discrete_laplacian[:, None] * uvw_j

    def update(self, aggr_out):
        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p):
        return p * r


def load_training_data(dataset_name, n_runs, log_dir, device):
    x_list = []
    y_list = []
    print('Load data ...')
    time.sleep(0.5)
    for run in trange(n_runs):
        x = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(x)
        y_list.append(y)
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device).squeeze()
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device).squeeze()
    print("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))
    x = []
    y = []

    return x_list, y_list, vnorm, ynorm


def plot_embedding_func_cluster_tracking(model, config, config_file, embedding_cluster, cmap, index_particles, indexes, type_list,
                                n_particle_types, n_particles, ynorm, epoch, log_dir, embedding_type, device):

    if embedding_type == 1:
        embedding = to_numpy(model.a.clone().detach())
        embedding = embedding[indexes.astype(int)]
        fig, ax = fig_init()
        for n in range(n_particle_types):
            pos = np.argwhere(type_list == n).squeeze().astype(int)
            plt.scatter(embedding[pos, 0], embedding[pos, 1], s=1, alpha=0.25)
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=78)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=78)
        plt.xlim(config.plotting.embedding_lim)
        plt.ylim(config.plotting.embedding_lim)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/all_embedding_{config_file}_{epoch}.tif", dpi=170.7)
        plt.close()
    else:
        fig, ax = fig_init()
        for k in trange(0, config.simulation.n_frames - 2):
            embedding = to_numpy(model.a[k * n_particles:(k + 1) * n_particles, :].clone().detach())
            for n in range(n_particle_types):
                plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], s=1,
                            color=cmap.color(n), alpha=0.025)
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=78)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=78)
        plt.xlim(config.plotting.embedding_lim)
        plt.ylim(config.plotting.embedding_lim)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/all_embedding_{config_file}_{epoch}.tif", dpi=170.7)
        plt.close()

    func_list, proj_interaction = analyze_edge_function_tracking(rr=[], vizualize=False, config=config,
                                                        model_MLP=model.lin_edge, model_a=model.a,
                                                        n_particles=n_particles, ynorm=ynorm,
                                                        indexes=indexes, type_list = type_list,
                                                        cmap=cmap, embedding_type = embedding_type, device=device)

    fig, ax = fig_init()
    proj_interaction = (proj_interaction - np.min(proj_interaction)) / (np.max(proj_interaction) - np.min(proj_interaction) + 1e-10)
    if embedding_type == 1:
        for n in range(n_particle_types):
            pos = np.argwhere(type_list == n).squeeze().astype(int)
            plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], s=1, alpha=0.25)
    else:
        for n in range(n_particle_types):
            plt.scatter(proj_interaction[index_particles[n], 0],
                        proj_interaction[index_particles[n], 1], color=cmap.color(n), s=1, alpha=0.25)
    plt.xlabel(r'UMAP 0', fontsize=78)
    plt.ylabel(r'UMAP 1', fontsize=78)
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/UMAP_{config_file}_{epoch}.tif", dpi=170.7)
    plt.close()

    embedding = to_numpy(model.a.clone().detach())
    if embedding_type == 1:
        embedding = embedding[indexes.astype(int)]
    else:
        embedding = embedding[0:n_particles]


    labels, n_clusters, new_labels = sparsify_cluster(config.training.cluster_method, proj_interaction, embedding,
                                                      config.training.cluster_distance_threshold, type_list,
                                                      n_particle_types, embedding_cluster)

    accuracy = metrics.accuracy_score(type_list, new_labels)

    fig, ax = fig_init()
    for n in np.unique(labels):
        pos = np.argwhere(labels == n).squeeze().astype(int)
        plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], s=1, alpha=0.25)

    return accuracy, n_clusters, new_labels


def plot_embedding_func_cluster_state(model, config, config_file, embedding_cluster, cmap, type_list, type_stack, id_list,
                                n_particle_types, ynorm, epoch, log_dir, device):

    n_frames = config.simulation.n_frames
    n_particles = config.simulation.n_particles

    fig, ax = fig_init()
    for k in range(0,len(type_list)):
        for n in range(n_particle_types):
            pos =torch.argwhere(type_list[k] == n)
            if len(pos) > 0:
                embedding = to_numpy(model.a[to_numpy(id_list[k][pos]).astype(int)].squeeze())
                plt.scatter(embedding[:, 0], embedding[:, 1], s=1, color=cmap.color(n), alpha=1)

    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=78)
    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=78)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/first_embedding_{config_file}_{epoch}.tif", dpi=170.7)
    plt.close()

    fig, ax = fig_init()
    func_list, true_type_list, short_model_a_list, proj_interaction = analyze_edge_function_state(rr=[], config=config,
                                                        model=model,
                                                        id_list=id_list, type_list=type_list, ynorm=ynorm,
                                                        cmap=cmap, visualize=True, device=device)
    np.save(f"./{log_dir}/results/function_{config_file}_{epoch}.npy", proj_interaction)

    fig, ax = fig_init()
    for n in range(n_particle_types):
        pos = np.argwhere(true_type_list == n).squeeze().astype(int)
        if len(pos)>0:
            plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], color=cmap.color(n), s=10, alpha=0.25)
    plt.xlabel(r'UMAP 0', fontsize=78)
    plt.ylabel(r'UMAP 1', fontsize=78)
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/UMAP_{config_file}_{epoch}.tif", dpi=170.7)
    plt.close()

    np.save(f"./{log_dir}/results/UMAP_{config_file}_{epoch}.npy", proj_interaction)

    embedding = proj_interaction
    labels, n_clusters, new_labels = sparsify_cluster_state(config.training.cluster_method, proj_interaction, embedding,
                                                      config.training.cluster_distance_threshold, true_type_list,
                                                      n_particle_types, embedding_cluster)

    fig, ax = fig_init()
    for n in range(n_particle_types):
        pos = np.argwhere(new_labels == n).squeeze().astype(int)
        if len(pos)>0:
            plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], color=cmap.color(n), s=10, alpha=0.25)
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.tight_layout()

    accuracy = metrics.accuracy_score(true_type_list, new_labels)

    median_center_list = []
    for n in range(n_clusters):
        pos = np.argwhere(new_labels == n).squeeze().astype(int)
        pos = np.array(pos)
        if pos.size > 0:
            median_center = short_model_a_list[pos, :]
            plt.scatter(to_numpy(short_model_a_list[pos,0]),to_numpy(short_model_a_list[pos,1]))
            median_center = torch.mean(median_center, dim=0)
            plt.scatter(to_numpy(median_center[0]), to_numpy(median_center[1]), s=100, color='black')
            median_center_list.append(median_center)
    median_center_list = torch.stack(median_center_list)
    median_center_list = median_center_list.to(dtype=torch.float32)

    distance = torch.sum((model.a[:, None, :] - median_center_list[None, :, :]) ** 2, dim=2)
    result = distance.min(dim=1)
    min_index = result.indices

    new_labels = to_numpy(min_index).astype(int)

    accuracy = metrics.accuracy_score(to_numpy(type_stack.squeeze()), new_labels)

    return accuracy, n_clusters, new_labels


def plot_embedding_func_cluster(model, config, config_file, embedding_cluster, cmap, index_particles, type_list,
                                n_particle_types, n_particles, ynorm, epoch, log_dir, alpha, device):

    fig, ax = fig_init()
    if config.training.do_tracking:
        embedding = to_numpy(model.a[0:n_particles])
    else:
        embedding = get_embedding(model.a, 1)
    if config.training.particle_dropout > 0:
        embedding = embedding[0:n_particles]

    if n_particle_types > 1000:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=to_numpy(x[:, 5]) / n_particles, s=10,
                    cmap=cc)
    else:
        for n in range(n_particle_types):
            pos = torch.argwhere(type_list == n)
            pos = to_numpy(pos)
            if len(pos) > 0:
                plt.scatter(embedding[pos, 0], embedding[pos, 1], s=100, alpha=alpha)
    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=78)
    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=78)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/first_embedding_{config_file}_{epoch}.tif", dpi=170.7)
    plt.close()

    fig, ax = fig_init()
    if 'PDE_N' in config.graph_model.signal_model_name:
        model_MLP_ = model.lin_phi
    else:
        model_MLP_ = model.lin_edge
    func_list, proj_interaction = analyze_edge_function(rr=[], vizualize=True, config=config, model_MLP=model_MLP_, model_a=model.a, type_list=to_numpy(type_list), n_particles=n_particles, dataset_number=1, ynorm=ynorm, cmap=cmap, device=device)
    plt.close()

    # trans = umap.UMAP(n_neighbors=100, n_components=2, init='spectral').fit(func_list_)
    # proj_interaction = trans.transform(func_list_)
    # tsne = TSNE(n_components=2, random_state=0)
    # proj_interaction =  tsne.fit_transform(func_list_)

    fig, ax = fig_init()
    proj_interaction = (proj_interaction - np.min(proj_interaction)) / (
                np.max(proj_interaction) - np.min(proj_interaction) + 1e-10)
    for n in range(n_particle_types):
        pos = torch.argwhere(type_list == n)
        pos = to_numpy(pos)
        if len(pos) > 0:
            plt.scatter(proj_interaction[pos, 0],
                        proj_interaction[pos, 1], color=cmap.color(n), s=200, alpha=0.1)
    plt.xlabel(r'UMAP 0', fontsize=78)
    plt.ylabel(r'UMAP 1', fontsize=78)
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/UMAP_{config_file}_{epoch}.tif", dpi=170.7)
    plt.close()

    config.training.cluster_distance_threshold = 0.01
    labels, n_clusters, new_labels = sparsify_cluster(config.training.cluster_method, proj_interaction, embedding,
                                                      config.training.cluster_distance_threshold, type_list,
                                                      n_particle_types, embedding_cluster)
    accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
    print(accuracy, n_clusters)

    model_a_ = model.a[1].clone().detach()
    for n in range(n_clusters):
        pos = np.argwhere(labels == n).squeeze().astype(int)
        pos = np.array(pos)
        if pos.size > 0:
            median_center = model_a_[pos, :]
            median_center = torch.median(median_center, dim=0).values
            model_a_[pos, :] = median_center
    with torch.no_grad():
        model.a[1] = model_a_.clone().detach()

    fig, ax = fig_init()
    embedding = get_embedding(model.a, 1)
    if n_particle_types > 1000:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=to_numpy(x[:, 5]) / n_particles, s=10,
                    cmap=cc)
    else:
        for n in range(n_particle_types):
            pos = torch.argwhere(type_list == n)
            pos = to_numpy(pos)
            if len(pos) > 0:
                plt.scatter(embedding[pos, 0], embedding[pos, 1], color=cmap.color(n),
                        s=100, alpha=0.1)
    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=78)
    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=78)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/embedding_{config_file}_{epoch}.tif", dpi=170.7)
    plt.close()

    return accuracy, n_clusters, new_labels


def plot_embedding(index, model_a, dataset_number, index_particles, n_particles, n_particle_types, epoch, it, fig, ax,
                   cmap, device):
    embedding = get_embedding(model_a, dataset_number)

    plt.text(-0.25, 1.1, f'{index}', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.title(r'Particle embedding', fontsize=12)
    for n in range(n_particle_types):
        plt.scatter(embedding[index_particles[n], 0],
                    embedding[index_particles[n], 1], color=cmap.color(n), s=0.1)
    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=12)
    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=12)
    plt.text(.05, .94, f'e: {epoch} it: {it}', ha='left', va='top', transform=ax.transAxes, fontsize=10)
    plt.text(.05, .86, f'N: {n_particles}', ha='left', va='top', transform=ax.transAxes, fontsize=10)
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)

    return embedding


def plot_function(bVisu, index, model_name, model_MLP, model_a, dataset_number, label, pos, max_radius, ynorm,
                  index_particles, n_particles, n_particle_types, epoch, it, fig, ax, cmap, device):
    # print(f'plot functions epoch:{epoch} it: {it}')

    plt.text(-0.25, 1.1, f'{index}', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.title(r'Interaction functions (model)', fontsize=12)
    func_list = []
    for n in range(n_particles):
        embedding_ = model_a[1, n, :] * torch.ones((1000, 2), device=device)

        match model_name:
            case 'PDE_A':
                in_features = torch.cat((pos[:, None] / max_radius, 0 * pos[:, None],
                                         pos[:, None] / max_radius, embedding_), dim=1)
            case 'PDE_A_bis':
                in_features = torch.cat((pos[:, None] / max_radius, 0 * pos[:, None],
                                         pos[:, None] / max_radius, embedding_, embedding_), dim=1)
            case 'PDE_B' | 'PDE_B_bis':
                in_features = torch.cat((pos[:, None] / max_radius, 0 * pos[:, None],
                                         pos[:, None] / max_radius, 0 * pos[:, None], 0 * pos[:, None],
                                         0 * pos[:, None], 0 * pos[:, None], embedding_), dim=1)
            case 'PDE_G':
                in_features = torch.cat((pos[:, None] / max_radius, 0 * pos[:, None],
                                         pos[:, None] / max_radius, 0 * pos[:, None], 0 * pos[:, None],
                                         0 * pos[:, None], 0 * pos[:, None], embedding_), dim=1)
            case 'PDE_GS':
                in_features = torch.cat((pos[:, None] / max_radius, embedding_), dim=1)
            case 'PDE_E':
                in_features = torch.cat((pos[:, None] / max_radius, 0 * pos[:, None],
                                         pos[:, None] / max_radius, embedding_, embedding_), dim=-1)

        with torch.no_grad():
            func = model_MLP(in_features.float())
        func = func[:, 0]
        func_list.append(func)
        if bVisu:
            plt.plot(to_numpy(pos),
                     to_numpy(func) * to_numpy(ynorm), color=cmap.color(label[n]), linewidth=1)
    func_list = torch.stack(func_list)
    func_list = to_numpy(func_list)
    if bVisu:
        plt.xlabel(r'$d_{ij} [a.u.]$', fontsize=12)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij} [a.u.]$', fontsize=12)
        plt.xticks(fontsize=10.0)
        plt.yticks(fontsize=10.0)
        # plt.ylim([-0.04, 0.03])
        plt.text(.05, .86, f'N: {n_particles // 50}', ha='left', va='top', transform=ax.transAxes, fontsize=10)
        plt.text(.05, .94, f'e: {epoch} it: {it}', ha='left', va='top', transform=ax.transAxes, fontsize=10)

    return func_list


def plot_umap(index, func_list, log_dir, n_neighbors, index_particles, n_particles, n_particle_types, embedding_cluster,
              epoch, it, fig, ax, cmap, device):
    # print(f'plot umap epoch:{epoch} it: {it}')
    plt.text(-0.25, 1.1, f'{index}', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    if False:  # os.path.exists(os.path.join(log_dir, f'proj_interaction_{epoch}.npy')):
        proj_interaction = np.load(os.path.join(log_dir, f'proj_interaction_{epoch}.npy'))
    else:
        new_index = np.random.permutation(func_list.shape[0])
        new_index = new_index[0:min(1000, func_list.shape[0])]
        trans = umap.UMAP(n_neighbors=n_neighbors, n_components=2, transform_queue_size=0).fit(func_list[new_index])
        proj_interaction = trans.transform(func_list)
    np.save(os.path.join(log_dir, f'proj_interaction_{epoch}.npy'), proj_interaction)
    plt.title(r'UMAP of $f(\ensuremath{\mathbf{a}}_i, d_{ij}$)', fontsize=12)

    labels, n_clusters = embedding_cluster.get(proj_interaction, 'distance')

    label_list = []
    for n in range(n_particle_types):
        tmp = labels[index_particles[n]]
        label_list.append(np.round(np.median(tmp)))
    label_list = np.array(label_list)
    new_labels = labels.copy()
    for n in range(n_particle_types):
        new_labels[labels == label_list[n]] = n
        plt.scatter(proj_interaction[index_particles[n], 0], proj_interaction[index_particles[n], 1],
                    color=cmap.color(n), s=1)

    plt.xlabel(r'UMAP 0', fontsize=12)
    plt.ylabel(r'UMAP 1', fontsize=12)

    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)
    plt.text(.05, .86, f'N: {n_particles}', ha='left', va='top', transform=ax.transAxes, fontsize=10)
    plt.text(.05, .94, f'e: {epoch} it: {it}', ha='left', va='top', transform=ax.transAxes, fontsize=10)

    return proj_interaction, new_labels, n_clusters


def plot_focused_on_cell(config, run, style, step, cell_id, device):

    dataset_name = config.dataset
    simulation_config = config.simulation
    model_config = config.graph_model
    training_config = config.training

    has_adjacency_matrix = (simulation_config.connectivity_file != '')
    has_mesh = (config.graph_model.mesh_model_name != '')
    only_mesh = (config.graph_model.particle_model_name == '') & has_mesh
    has_ghost = config.training.n_ghosts > 0
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    n_nodes = simulation_config.n_nodes
    n_runs = training_config.n_runs
    n_frames = simulation_config.n_frames
    delta_t = simulation_config.delta_t
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    dimension = simulation_config.dimension
    has_siren = 'siren' in model_config.field_type
    has_siren_time = 'siren_with_time' in model_config.field_type
    has_field = ('PDE_ParticleField' in config.graph_model.particle_model_name)

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    files = glob.glob(f"./{log_dir}/tmp_recons/*")
    for f in files:
        os.remove(f)

    print('Load data ...')

    x_list = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device)


    mass_time_series = get_time_series(x_list, cell_id, feature='mass')
    vx_time_series = get_time_series(x_list, cell_id, feature='velocity_x')
    vy_time_series = get_time_series(x_list, cell_id, feature='velocity_y')
    v_time_series = np.sqrt(vx_time_series ** 2 + vy_time_series ** 2)
    stage_time_series = get_time_series(x_list, cell_id, feature="stage")
    stage_time_series_color = ["blue" if i == 0 else "orange" if i == 1 else "green" if i == 2 else "pink" for i in stage_time_series]


    for it in trange(0,n_frames,step):

        x = x_list[it].clone().detach()

        T1 = x[:, 5:6].clone().detach()
        H1 = x[:, 6:8].clone().detach()
        X1 = x[:, 1:3].clone().detach()

        index_particles = get_index_particles(x, n_particle_types, dimension)

        pos_cell = torch.argwhere(x[:,0] == cell_id)

        if len(pos_cell)>0:

            if 'latex' in style:
                plt.rcParams['text.usetex'] = True
                rc('font', **{'family': 'serif', 'serif': ['Palatino']})

            if 'color' in style:

                # matplotlib.use("Qt5Agg")
                matplotlib.rcParams['savefig.pad_inches'] = 0
                fig = plt.figure(figsize=(24, 12))
                ax = fig.add_subplot(1, 2, 1)
                ax.xaxis.get_major_formatter()._usetex = False
                ax.yaxis.get_major_formatter()._usetex = False
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                index_particles = []
                for n in range(n_particle_types):
                    pos = torch.argwhere((T1.squeeze() == n) & (H1[:, 0].squeeze() == 1))
                    pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                    index_particles.append(pos)
                    # plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                    #             s=marker_size, color=cmap.color(n))

                    size = set_size(x, index_particles[n], 10)

                    plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                                s=size, color=cmap.color(n))
                    
                dead_cell = np.argwhere(to_numpy(H1[:, 0]) == 0)
                if len(dead_cell) > 0:
                    plt.scatter(to_numpy(X1[dead_cell[:, 0].squeeze(), 0]), to_numpy(X1[dead_cell[:, 0].squeeze(), 1]),
                                s=2, color='k', alpha=0.5)
                if 'latex' in style:
                    plt.xlabel(r'$x$', fontsize=78)
                    plt.ylabel(r'$y$', fontsize=78)
                    plt.xticks(fontsize=48.0)
                    plt.yticks(fontsize=48.0)

                elif 'frame' in style:
                    plt.xlabel('x', fontsize=13)
                    plt.ylabel('y', fontsize=16)
                    plt.xticks(fontsize=16.0)
                    plt.yticks(fontsize=16.0)
                    ax.tick_params(axis='both', which='major', pad=15)
                    plt.text(0, 1.05,
                             f'frame {it}, {int(n_particles_alive)} alive particles ({int(n_particles_dead)} dead), {edge_index.shape[1]} edges  ',
                             ha='left', va='top', transform=ax.transAxes, fontsize=16)

                plt.xticks([])
                plt.yticks([])

                center_x = to_numpy(x[pos_cell, 1])
                center_y = to_numpy(x[pos_cell, 2])
                plt.xlim([center_x - 0.1, center_x + 0.1])
                plt.ylim([center_y - 0.1, center_y + 0.1])

                ax = fig.add_subplot(2, 2, 2)
                plt.plot(mass_time_series, color='k', ls="--")

                plt.scatter([i for i in range(it)], mass_time_series[0:it], color=stage_time_series_color[0:it], s=15)
                # plt.plot(mass_time_series[0:it], color = color,linewidth=3)
                plt.ylim(0, max(mass_time_series) + 50)

                ax = fig.add_subplot(2, 2, 4)
                plt.plot(v_time_series, color='k', ls="--")
                plt.plot(v_time_series[0:it], color = 'red',linewidth=4)

                num = f"{it:06}"

                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_recons/cell_{cell_id}_frame_{num}.tif", dpi=80)
                plt.close()


def plot_generated(config, run, style, step, device):

    dataset_name = config.dataset
    simulation_config = config.simulation
    model_config = config.graph_model
    training_config = config.training

    has_adjacency_matrix = (simulation_config.connectivity_file != '')
    has_mesh = (config.graph_model.mesh_model_name != '')
    only_mesh = (config.graph_model.particle_model_name == '') & has_mesh
    has_ghost = config.training.n_ghosts > 0
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    n_nodes = simulation_config.n_nodes
    n_runs = training_config.n_runs
    n_frames = simulation_config.n_frames
    delta_t = simulation_config.delta_t
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    dimension = simulation_config.dimension
    has_siren = 'siren' in model_config.field_type
    has_siren_time = 'siren_with_time' in model_config.field_type
    has_field = ('PDE_ParticleField' in config.graph_model.particle_model_name)

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    files = glob.glob(f"./{log_dir}/tmp_recons/*")
    for f in files:
        os.remove(f)

    os.makedirs(os.path.join(log_dir, 'generated_bw'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'generated_color'), exist_ok=True)

    files = glob.glob(f"./{log_dir}/generated_bw/*")
    for f in files:
        os.remove(f)

    files = glob.glob(f"./{log_dir}/generated_color/*")
    for f in files:
        os.remove(f)

    print('Load data ...')

    x_list = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device)


    for it in trange(0,n_frames,step):

        x = x_list[it].clone().detach()

        T1 = x[:, 5:6].clone().detach()
        H1 = x[:, 6:8].clone().detach()
        X1 = x[:, 1:3].clone().detach()

        if 'latex' in style:
            plt.rcParams['text.usetex'] = True
            rc('font', **{'family': 'serif', 'serif': ['Palatino']})


        if 'voronoi' in style:
            matplotlib.use("Qt5Agg")
            matplotlib.rcParams['savefig.pad_inches'] = 0

            vor, vertices_pos, vertices_per_cell, all_points = get_vertices(points=X1, device=device)

            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1, 1, 1)
            plt.xticks([])
            plt.yticks([])
            index_particles = []

            voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=1, line_alpha=0.5, point_size=0)

            if 'color' in style:
                for n in range(n_particle_types):
                    pos = torch.argwhere((T1.squeeze() == n) & (H1[:, 0].squeeze() == 1))
                    pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                    index_particles.append(pos)

                    size = set_size(x, index_particles[n], 10) / 10

                    patches = []
                    for i in index_particles[n]:
                        cell = vertices_per_cell[i]
                        vertices = to_numpy(vertices_pos[cell, :])
                        patches.append(Polygon(vertices, closed=True))

                    pc = PatchCollection(patches, alpha=0.4, facecolors=cmap.color(n))
                    ax.add_collection(pc)
                    if 'center' in style:
                        plt.scatter(to_numpy(X1[index_particles[n], 0]), to_numpy(X1[index_particles[n], 1]), s=size,
                                    color=cmap.color(n))

            if 'vertices' in style:
                plt.scatter(to_numpy(vertices_pos[:, 0]), to_numpy(vertices_pos[:, 1]), s=5, color='k')

            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.tight_layout()

            num = f"{it:06}"
            if 'color' in style:
                plt.savefig(f"./{log_dir}/generated_color/frame_{num}.tif", dpi=85.35)
            else:
                plt.savefig(f"./{log_dir}/generated_bw/frame_{num}.tif", dpi=85.35)
            plt.close()


        else:

            matplotlib.rcParams['savefig.pad_inches'] = 0
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1, 1, 1)
            ax.xaxis.get_major_formatter()._usetex = False
            ax.yaxis.get_major_formatter()._usetex = False
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            index_particles = []
            for n in range(n_particle_types):
                pos = torch.argwhere((T1.squeeze() == n) & (H1[:, 0].squeeze() == 1))
                pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                index_particles.append(pos)
                # plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                #             s=marker_size, color=cmap.color(n))

                size = set_size(x, index_particles[n], 10) / 10

                plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                            s=40, color=cmap.color(n))
            dead_cell = np.argwhere(to_numpy(H1[:, 0]) == 0)
            if len(dead_cell) > 0:
                plt.scatter(to_numpy(X1[dead_cell[:, 0].squeeze(), 0]), to_numpy(X1[dead_cell[:, 0].squeeze(), 1]),
                            s=2, color='k', alpha=0.5)
            if 'latex' in style:
                plt.xlabel(r'$x$', fontsize=78)
                plt.ylabel(r'$y$', fontsize=78)
                plt.xticks(fontsize=48.0)
                plt.yticks(fontsize=48.0)
            elif 'frame' in style:
                plt.xlabel('x', fontsize=13)
                plt.ylabel('y', fontsize=16)
                plt.xticks(fontsize=16.0)
                plt.yticks(fontsize=16.0)
                ax.tick_params(axis='both', which='major', pad=15)
                plt.text(0, 1.05,
                         f'frame {it}, {int(n_particles_alive)} alive particles ({int(n_particles_dead)} dead), {edge_index.shape[1]} edges  ',
                         ha='left', va='top', transform=ax.transAxes, fontsize=16)
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.tight_layout()
            num = f"{it:06}"
            plt.savefig(f"./{log_dir}/generated_color/frame_{num}.tif", dpi=80)
            plt.close()

            matplotlib.rcParams['savefig.pad_inches'] = 0
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1, 1, 1)
            ax.xaxis.get_major_formatter()._usetex = False
            ax.yaxis.get_major_formatter()._usetex = False
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
            index_particles = []
            for n in range(n_particle_types):
                pos = torch.argwhere((T1.squeeze() == n) & (H1[:, 0].squeeze() == 1))
                pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                index_particles.append(pos)
                # plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                #             s=marker_size, color=cmap.color(n))
                size = set_size(x, index_particles[n], 10)
                plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                            s=size/10, color='k')
            dead_cell = np.argwhere(to_numpy(H1[:, 0]) == 0)
            if len(dead_cell) > 0:
                plt.scatter(to_numpy(X1[dead_cell[:, 0].squeeze(), 0]), to_numpy(X1[dead_cell[:, 0].squeeze(), 1]),
                            s=2, color='k', alpha=0.5)
            if 'latex' in style:
                plt.xlabel(r'$x$', fontsize=78)
                plt.ylabel(r'$y$', fontsize=78)
                plt.xticks(fontsize=48.0)
                plt.yticks(fontsize=48.0)
            elif 'frame' in style:
                plt.xlabel('x', fontsize=13)
                plt.ylabel('y', fontsize=16)
                plt.xticks(fontsize=16.0)
                plt.yticks(fontsize=16.0)
                ax.tick_params(axis='both', which='major', pad=15)
                plt.text(0, 1.05,
                         f'frame {it}, {int(n_particles_alive)} alive particles ({int(n_particles_dead)} dead), {edge_index.shape[1]} edges  ',
                         ha='left', va='top', transform=ax.transAxes, fontsize=16)
            plt.xticks([])
            plt.yticks([])
            plt.xlim([0,1])
            plt.ylim([0,1])
            plt.tight_layout()
            num = f"{it:06}"
            plt.savefig(f"./{log_dir}/generated_bw/frame_{num}.tif", dpi=80)
            plt.close()


def plot_confusion_matrix(index, true_labels, new_labels, n_particle_types, epoch, it, fig, ax):
    # print(f'plot confusion matrix epoch:{epoch} it: {it}')
    plt.text(-0.25, 1.1, f'{index}', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    confusion_matrix = metrics.confusion_matrix(true_labels, new_labels)  # , normalize='true')
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    if n_particle_types > 8:
        cm_display.plot(ax=fig.gca(), cmap='Blues', include_values=False, colorbar=False)
    else:
        cm_display.plot(ax=fig.gca(), cmap='Blues', include_values=True, values_format='d', colorbar=False)
    accuracy = metrics.accuracy_score(true_labels, new_labels)
    plt.title(f'accuracy: {np.round(accuracy, 2)}', fontsize=12)
    # print(f'accuracy: {np.round(accuracy,3)}')
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)
    plt.xlabel(r'Predicted label', fontsize=12)
    plt.ylabel(r'True label', fontsize=12)

    return accuracy


def plot_cell_rates(config, device, log_dir, n_particle_types, type_list, x_list, new_labels, cmap, logger):

    n_frames = config.simulation.n_frames
    delta_t = config.simulation.delta_t

    cell_cycle_length = np.array(config.simulation.cell_cycle_length)
    if len(cell_cycle_length) == 1:
        cell_cycle_length = to_numpy(torch.load(f'graphs_data/graphs_{config.dataset}/cycle_length.pt', map_location=device))


    print('plot cell rates ...')
    N_cells_alive = np.zeros((n_frames, n_particle_types))
    N_cells_dead = np.zeros((n_frames, n_particle_types))

    if os.path.exists(f"./{log_dir}/results/x_.npy"):
        x_ = np.load(f"./{log_dir}/results/x_.npy")
        N_cells_alive = np.load(f"./{log_dir}/results/cell_alive.npy")
        N_cells_dead = np.load(f"./{log_dir}/results/cell_dead.npy")
    else:
        for it in trange(n_frames):

            x = x_list[0][it].clone().detach()
            particle_index = to_numpy(x[:, 0:1]).astype(int)
            x[:, 5:6] = torch.tensor(new_labels[particle_index], device=device)
            if it == 0:
                x_=x_list[0][it].clone().detach()
            else:
                x_=torch.concatenate((x_,x),axis=0)

            for k in range(n_particle_types):
                pos = torch.argwhere((x[:, 5:6] == k) & (x[:, 6:7] == 1))
                N_cells_alive[it, k] = pos.shape[0]
                pos = torch.argwhere((x[:, 5:6] == k) & (x[:, 6:7] == 0))
                N_cells_dead[it, k] = pos.shape[0]

        x_list=[]
        x_ = to_numpy(x_)

        print('save data ...')

        np.save(f"./{log_dir}/results/cell_alive.npy", N_cells_alive)
        np.save(f"./{log_dir}/results/cell_dead.npy", N_cells_dead)
        np.save(f"./{log_dir}/results/x_.npy", x_)

    print('plot results ...')

    last_frame_growth = np.argwhere(np.diff(N_cells_alive[:, 0], axis=0))
    last_frame_growth = last_frame_growth[-1] - 1
    N_cells_alive = N_cells_alive[0:int(last_frame_growth), :]
    N_cells_dead = N_cells_dead[0:int(last_frame_growth), :]

    fig, ax = fig_init()
    for k in range(n_particle_types):
        plt.plot(np.arange(last_frame_growth), N_cells_alive[:, k], color=cmap.color(k), linewidth=4,
                 label=f'Cell type {k} alive')
    plt.xlabel(r'Frame', fontsize=64)
    plt.ylabel(r'Number of cells', fontsize=64)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/cell_alive_{config_file}.tif", dpi=300)
    plt.close()

    fig, ax = fig_init()
    for k in range(n_particle_types):
        plt.plot(np.arange(last_frame_growth), N_cells_dead[:, k], color=cmap.color(k), linewidth=4,
                 label=f'Cell type {k} dead')
    plt.xlabel(r'Frame', fontsize=78)
    plt.ylabel(r'Number of dead cells', fontsize=78)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/cell_dead_{config_file}.tif", dpi=300)
    plt.close()

    #         6,7 H1 cell status dim=2  H1[:,0] = cell alive flag, alive : 0 , death : 0 , H1[:,1] = cell division flag, dividing : 1
    #         8 A1 cell age dim=1

    division_list = {}
    for n in np.unique(new_labels):
        division_list[n] = []
    for n in trange(len(type_list)):
        pos = np.argwhere(x_[:, 0:1] == n)
        if len(pos)>0:
            division_list[new_labels[n]].append(len(pos)* delta_t)

    reconstructed_cell_cycle_length = np.zeros(n_particle_types)
    for k in range(n_particle_types):
        print(f'Cell type {k} division rate: {np.mean(division_list[k])}+/-{np.std(division_list[k])}')
        logger.info(f'Cell type {k} division rate: {np.mean(division_list[k])}+/-{np.std(division_list[k])}')
        reconstructed_cell_cycle_length[k] = np.mean(division_list[k])

    x_data = cell_cycle_length
    y_data = reconstructed_cell_cycle_length.squeeze()
    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
    residuals = y_data - linear_model(x_data, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f'R^2$: {np.round(r_squared, 3)}  slope: {np.round(lin_fit[0], 2)}')
    logger.info(f'R^2$: {np.round(r_squared, 3)}  slope: {np.round(lin_fit[0], 2)}')

    fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
    plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
    plt.scatter(cell_cycle_length,reconstructed_cell_cycle_length, color=cmap.color(np.arange(n_particle_types)), s=200)
    plt.xlabel(r'True cell cycle length', fontsize=54)
    plt.ylabel(r'Learned cell cycle length', fontsize=54)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/cell_cycle_length_{config_file}.tif", dpi=170)
    plt.close()


    division_list = {}
    for n in np.unique(new_labels):
        division_list[n] = []
    for n in trange(n_frames):
        x = x_list[0][n].clone().detach()
        pos = torch.argwhere(x[:, 7:8] == 0)
        if pos.shape[0]>1:
            x = x[pos]
            for x_ in x:
                division_list[x_[5]].append(x_[8])


def plot_attraction_repulsion(config_file, epoch_list, log_dir, logger, device):

    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    dimension = config.simulation.dimension
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    max_radius = config.simulation.max_radius
    cmap = CustomColorMap(config=config)
    n_runs = config.training.n_runs
    has_particle_dropout = config.training.particle_dropout > 0
    dataset_name = config.dataset

    embedding_cluster = EmbeddingCluster(config)

    x_list, y_list, vnorm, ynorm = load_training_data(dataset_name, n_runs, log_dir, device)
    logger.info("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))
    x = x_list[1][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    n_particles = x.shape[0]

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    for epoch in epoch_list:

        net = f"./log/try_{config_file}/models/best_model_with_1_graphs_{epoch}.pt"
        print(f'network: {net}')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()

        model_a_first = model.a.clone().detach()

        config.training.cluster_method = 'distance_plot'
        config.training.cluster_distance_threshold = 0.01
        alpha=0.1
        accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config, config_file, embedding_cluster,
                                                                       cmap, index_particles, type_list,
                                                                       n_particle_types, n_particles, ynorm, epoch,
                                                                       log_dir, alpha, device)
        print(
            f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
        logger.info(
            f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()
        config.training.cluster_method = 'distance_embedding'
        config.training.cluster_distance_threshold = 0.01
        alpha = 0.1
        accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config, config_file, embedding_cluster,
                                                                       cmap, index_particles, type_list,
                                                                       n_particle_types, n_particles, ynorm, epoch,
                                                                       log_dir, alpha, device)
        print(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
        logger.info(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')

        fig, ax = fig_init()
        p = torch.load(f'graphs_data/graphs_{dataset_name}/model_p.pt', map_location=device)
        rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
        rmserr_list = []
        for n in range(int(n_particles * (1 - config.training.particle_dropout))):
            embedding_ = model_a_first[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding_), dim=1)
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
                func = func[:, 0]
            true_func = model.psi(rr, p[to_numpy(type_list[n]).astype(int)].squeeze(),
                                  p[to_numpy(type_list[n]).astype(int)].squeeze())
            rmserr_list.append(torch.sqrt(torch.mean((func * ynorm - true_func.squeeze()) ** 2)))
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=8, alpha=0.1)
        plt.xlabel(r'$d_{ij}$', fontsize=78)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=78)
        plt.xlim([0, max_radius])
        plt.ylim(config.plotting.ylim)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/func_all_{config_file}_{epoch}.tif", dpi=170.7)
        rmserr_list = torch.stack(rmserr_list)
        rmserr_list = to_numpy(rmserr_list)
        print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
        logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
        plt.close()

        fig, ax = fig_init()
        plots = []
        plots.append(rr)
        for n in range(n_particle_types):
            plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n), linewidth=8)
            plots.append(model.psi(rr, p[n], p[n]).squeeze())
        plt.xlim([0, max_radius])
        plt.ylim(config.plotting.ylim)
        plt.xlabel(r'$d_{ij}$', fontsize=78)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/true_func_{config_file}.tif", dpi=170.7)
        plt.close()


def plot_cell_state(config_file, epoch_list, log_dir, logger, device):

    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    dimension = config.simulation.dimension
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    max_radius = config.simulation.max_radius
    cmap = CustomColorMap(config=config)
    n_runs = config.training.n_runs
    n_frames = config.simulation.n_frames
    do_tracking = config.training.do_tracking
    has_cell_division = config.simulation.has_cell_division

    embedding_cluster = EmbeddingCluster(config)

    x_list, y_list, vnorm, ynorm = load_training_data(dataset_name, n_runs, log_dir, device)
    logger.info("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))

    type_stack = torch.stack(x_list[1])[:,:,5]
    type_stack = torch.reshape(type_stack, ((n_frames + 1)* n_particles,1))

    n_particles_max = 0
    id_list = []
    type_list=[]
    for k in range(n_frames+1):
        type = x_list[1][k][:, 5]
        type_list.append(type)
        ids = x_list[1][k][:, -1]
        id_list.append(ids)
        n_particles_max += len(type)
    config.simulation.n_particles_max = n_particles_max

    config.training.use_hot_encoding = True
    model, bc_pos, bc_dpos = choose_training_model(config, device)

    for epoch in epoch_list:

        net = f"./log/try_{config_file}/models/best_model_with_1_graphs_{epoch}.pt"
        print(f'network: {net}')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()

        fig = plt.figure(figsize=(10, 10))
        plt.scatter(to_numpy(model.a[:, 0]), to_numpy(model.a[:, 1]), c=to_numpy(type_stack), s=1, cmap='tab10')

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(to_numpy(model.a[:, 0]), to_numpy(model.a[:, 1]), to_numpy(model.a[:, 2]), c=to_numpy(type_stack), s=1, cmap='tab20')

        accuracy, n_clusters, new_labels = plot_embedding_func_cluster_state(model, config, config_file, embedding_cluster,
                                                                       cmap, type_list, type_stack, id_list,
                                                                       n_particle_types, ynorm, epoch,
                                                                       log_dir, device)


        print(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
        logger.info(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')

        fig, ax = fig_init()
        plots = []
        p = torch.load(f'graphs_data/graphs_{dataset_name}/model_p.pt', map_location=device)
        rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
        plots.append(rr)
        for n in range(n_particle_types):
            plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n), linewidth=8)
            plots.append(model.psi(rr, p[n], p[n]).squeeze())
        plt.xlim([0, max_radius])
        plt.ylim(config.plotting.ylim)
        plt.xlabel(r'$d_{ij}$', fontsize=78)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/true_func_{config_file}.tif", dpi=170.7)
        plt.close()

        learned_time_series = np.reshape(new_labels, (n_frames + 1, n_particles))
        GT_time_series = np.reshape(to_numpy(type_stack), (n_frames + 1, n_particles))

        fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
        plt.imshow(np.rot90(GT_time_series[:,0:80]), aspect='auto', cmap='tab10',vmin=0, vmax=2)
        plt.xlabel('frame', fontsize=78)
        plt.ylabel('cell_id', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/true_kinograph_{config_file}.tif", dpi=170.7)
        plt.close()

        fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
        plt.imshow(np.rot90(learned_time_series[:,0:80]), aspect='auto', cmap='tab10',vmin=0, vmax=2)
        plt.xlabel('frame', fontsize=78)
        plt.ylabel('cell_id', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/learned_kinograph_{config_file}.tif", dpi=170.7)
        plt.close()

        fig = plt.figure(figsize=(10, 10))

        GT = GT_time_series
        time_series = learned_time_series

        new_time_series = 0 * time_series
        accuracy_list = []
        for k in trange(n_particles):
            input = time_series[:,k]
            output = median_filter(input, size=5, mode='nearest')
            new_time_series[:,k] = output
            accuracy = metrics.accuracy_score(GT[:,k], output)
            accuracy_list.append(accuracy)

        accuracy = np.array(accuracy_list)
        print(f'accuracy: {np.mean(accuracy)} +/- {np.std(accuracy)}')

        fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
        plt.imshow(np.rot90(new_time_series), aspect='auto', cmap='tab10',vmin=0, vmax=2)
        plt.xlabel('frame', fontsize=78)
        plt.ylabel('cell_id', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/filtered_learned_kinograph_{config_file}.tif", dpi=170.7)


def plot_attraction_repulsion_tracking(config_file, epoch_list, log_dir, logger, device):

    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    dimension = config.simulation.dimension
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    min_radius = config.simulation.min_radius
    max_radius = config.simulation.max_radius
    cmap = CustomColorMap(config=config)
    n_runs = config.training.n_runs
    n_frames = config.simulation.n_frames
    delta_t = config.simulation.delta_t
    sequence_length = len(config.training.sequence)
    has_state = (config.simulation.state_type != ['discrete'])

    embedding_cluster = EmbeddingCluster(config)

    x_list, y_list, vnorm, ynorm = load_training_data(dataset_name, n_runs, log_dir, device)
    logger.info("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))
    x = x_list[1][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    type_list_first = type_list.clone().detach()

    index_l = []
    index = 0
    for k in range(n_frames):
        new_index = torch.arange(index, index + n_particles)
        index_l.append(new_index)
        x_list[1][k][:, 0] = new_index
        index += n_particles

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    accuracy_list_=[]
    tracking_index_list_=[]
    tracking_errors_list_=[]

    for epoch in epoch_list:
        print('')
        logger.info('')
        pos = epoch.find('_')
        if pos>0:
            epoch_ = epoch[0:pos]
        else:
            epoch_ = epoch
        embedding_type = int(epoch_)%sequence_length
        print(f'{epoch}, {epoch_}, {embedding_type}')
        logger.info(f'{epoch}, {epoch_}, {embedding_type}')

        net = f"./log/try_{config_file}/models/best_model_with_1_graphs_{epoch}.pt"
        print(f'network: {net}')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()

        fig = plt.figure(figsize=(8, 8))
        tracking_index = 0
        tracking_index_list = []
        for k in trange(n_frames):
            x = x_list[1][k].clone().detach()
            distance = torch.sum(bc_dpos(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, dim=2)
            adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
            edges = adj_t.nonzero().t().contiguous()
            dataset = data.Data(x=x[:, :], edge_index=edges)

            pred = model(dataset, training=True, vnorm=vnorm, phi=torch.zeros(1, device=device))

            x_next = x_list[1][k + 1]
            x_next = x_next[:, 1:3].clone().detach()
            x_pred = (x[:, 1:3] + delta_t * pred)

            distance = torch.sum(bc_dpos(x_pred[:, None, :] - x_next[None, :, :]) ** 2, dim=2)
            result = distance.min(dim=1)
            min_value = result.values
            min_index = result.indices

            true_index = np.arange(len(min_index))
            reconstructed_index = to_numpy(min_index)
            for n in range(n_particle_types):
                plt.scatter(true_index[index_particles[n]], reconstructed_index[index_particles[n]], s=1, color=cmap.color(n), alpha=0.05)

            tracking_index += np.sum((to_numpy(min_index) - np.arange(len(min_index)) == 0)) / n_frames / n_particles * 100
            tracking_index_list.append(np.sum((to_numpy(min_index) - np.arange(len(min_index)) == 0)))
            x_list[1][k + 1][min_index, 0:1] = x_list[1][k][:, 0:1].clone().detach()

        x_ = torch.stack(x_list[1])
        x_ = torch.reshape(x_, (x_.shape[0] * x_.shape[1], x_.shape[2]))
        x_ = x_[0:(n_frames - 1) * n_particles]
        indexes = np.unique(to_numpy(x_[:, 0]))

        plt.xlabel(r'True particle index', fontsize=32)
        plt.ylabel(r'Particle index in next frame', fontsize=32)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/proxy_tracking_{config_file}_{epoch}.tif", dpi=170.7)
        plt.close()
        print(f'tracking index: {np.round(tracking_index,3)}')
        logger.info(f'tracking index: {np.round(tracking_index,3)}')
        print(f'{len(indexes)} tracks')
        logger.info(f'{len(indexes)} tracks')

        tracking_index_list_.append(tracking_index)

        tracking_index_list = np.array(tracking_index_list)
        tracking_index_list = n_particles - tracking_index_list

        fig,ax = fig_init(formatx='%.0f', formaty='%.0f')
        plt.plot(np.arange(n_frames), tracking_index_list, color='k', linewidth=2)
        plt.ylabel(r'tracking errors', fontsize=78)
        plt.xlabel(r'frame', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/tracking_error_{config_file}_{epoch}.tif", dpi=170.7)
        plt.close()

        print(f'tracking errors: {np.sum(tracking_index_list)}')
        logger.info(f'tracking errors: {np.sum(tracking_index_list)}')

        tracking_errors_list_.append(np.sum(tracking_index_list))

        if embedding_type==1:
            type_list = to_numpy(x_[indexes,5])
        else:
            type_list = to_numpy(type_list_first)

        config.training.cluster_distance_threshold = 0.1
        model_a_first = model.a.clone().detach()
        alpha = 0.1
        accuracy, n_clusters, new_labels = plot_embedding_func_cluster_tracking(model, config, config_file, embedding_cluster, cmap, index_particles, indexes, type_list,
                                n_particle_types, n_particles, ynorm, epoch, log_dir, embedding_type, alpha, device)
        print(
            f'accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
        logger.info(
            f'accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')

        accuracy_list_.append(accuracy)

        if embedding_type==1:
            fig, ax = fig_init()
            p = torch.load(f'graphs_data/graphs_{dataset_name}/model_p.pt', map_location=device)
            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
            rmserr_list = []
            for n, k in enumerate(indexes):
                embedding_ = model_a_first[int(k), :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         rr[:, None] / max_radius, embedding_), dim=1)
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                func = func[:, 0]
                true_func = model.psi(rr, p[int(type_list[int(n)])].squeeze(),
                                      p[int(type_list[int(n)])].squeeze())
                rmserr_list.append(torch.sqrt(torch.mean((func - true_func.squeeze()) ** 2)))
                plt.plot(to_numpy(rr),
                         to_numpy(func),
                         color=cmap.color(int(type_list[int(n)])), linewidth=2, alpha=0.1)
            plt.xlabel(r'$d_{ij}$', fontsize=78)
            plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=78)
            plt.xlim([0, max_radius])
            plt.ylim(config.plotting.ylim)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/func_all_{config_file}_{epoch}.tif", dpi=170.7)
            rmserr_list = torch.stack(rmserr_list)
            rmserr_list = to_numpy(rmserr_list)
            print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
            logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
            plt.close()
        else:
            fig, ax = fig_init()
            p = torch.load(f'graphs_data/graphs_{dataset_name}/model_p.pt', map_location=device)
            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
            rmserr_list = []
            for n in range(int(n_particles * (1 - config.training.particle_dropout))):
                embedding_ = model_a_first[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         rr[:, None] / max_radius, embedding_), dim=1)
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                func = func[:, 0]
                true_func = model.psi(rr, p[int(type_list[n,0])],p[int(type_list[n,0])])
                rmserr_list.append(torch.sqrt(torch.mean((func - true_func.squeeze()) ** 2)))
                plt.plot(to_numpy(rr), to_numpy(func), color=cmap.color(int(type_list[n,0])), linewidth=2, alpha=0.1)
            plt.xlabel(r'$d_{ij}$', fontsize=78)
            plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=78)
            plt.xlim([0, max_radius])
            plt.ylim(config.plotting.ylim)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/func_all_{config_file}_{epoch}.tif", dpi=170.7)
            rmserr_list = torch.stack(rmserr_list)
            rmserr_list = to_numpy(rmserr_list)
            print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
            logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
            plt.close()

        fig, ax = fig_init()
        for n in range(n_particle_types):
            plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n), linewidth=8)
        plt.xlim([0, max_radius])
        plt.ylim(config.plotting.ylim)
        plt.xlabel(r'$d_{ij}$', fontsize=78)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/true_func_{config_file}.tif", dpi=170.7)
        plt.close()

        fig, ax = fig_init()
        plt.plot(tracking_index_list_, color='k', linewidth=2)
        plt.ylabel(r'tracking_index', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/tracking_index_list_{config_file}.tif", dpi=170.7)
        fig, ax = fig_init()
        plt.plot(accuracy_list_, color='k', linewidth=2)
        plt.ylabel(r'accuracy', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/accuracy_list_{config_file}.tif", dpi=170.7)
        fig, ax = fig_init()
        plt.plot(tracking_errors_list_, color='k', linewidth=2)
        plt.ylabel(r'tracking_errors', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/tracking_errors_list_{config_file}.tif", dpi=170.7)


def plot_attraction_repulsion_asym(config_file, epoch_list, log_dir, logger, device):

    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    cmap = CustomColorMap(config=config)
    embedding_cluster = EmbeddingCluster(config)
    n_runs = config.training.n_runs

    x_list, y_list, vnorm, ynorm = load_training_data(dataset_name, n_runs, log_dir, device)
    logger.info("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))
    model, bc_pos, bc_dpos = choose_training_model(config, device)

    x = x_list[1][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)

    for epoch in epoch_list:

        net = f"./log/try_{config_file}/models/best_model_with_1_graphs_{epoch}.pt"
        print(f'network: {net}')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()

        config.training.cluster_method = 'distance_embedding'
        config.training.cluster_distance_threshold = 0.01
        alpha = 0.1
        accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config, config_file, embedding_cluster,
                                                                       cmap, index_particles, type_list,
                                                                       n_particle_types, n_particles, ynorm, epoch,
                                                                       log_dir, alpha, device)
        print(
            f'final result     accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
        logger.info(
            f'final result     accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')

        x = x_list[0][100].clone().detach()
        index_particles = get_index_particles(x, n_particle_types, dimension)
        type_list = to_numpy(get_type_list(x, dimension))
        distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
        adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
        edges = adj_t.nonzero().t().contiguous()
        indexes = np.random.randint(0, edges.shape[1], 5000)
        edges = edges[:, indexes]

        fig, ax = fig_init()
        rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
        func_list = []
        for n in trange(edges.shape[1]):
            embedding_1 = model.a[1, edges[0, n], :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                  device=device)
            embedding_2 = model.a[1, edges[1, n], :] * torch.ones((1000, config.graph_model.embedding_dim),
                                                                  device=device)
            type = type_list[to_numpy(edges[0, n])].astype(int)
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding_1, embedding_2), dim=1)
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            func = func[:, 0]
            func_list.append(func)
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color(type), linewidth=8)
        plt.xlabel(r'$d_{ij}$', fontsize=78)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, \ensuremath{\mathbf{a}}_j, d_{ij})$', fontsize=78)
        plt.ylim(config.plotting.ylim)
        plt.xlim([0, max_radius])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/func_{config_file}_{epoch}.tif", dpi=170.7)
        plt.close()

        fig, ax = fig_init()
        p = torch.load(f'graphs_data/graphs_{dataset_name}/model_p.pt', map_location=device)
        true_func = []
        for n in range(n_particle_types):
            for m in range(n_particle_types):
                true_func.append(model.psi(rr, p[n, m].squeeze(), p[n, m].squeeze()))
                plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n,m], p[n,m]).squeeze()), color=cmap.color(n), linewidth=8)
        plt.xlabel(r'$d_{ij}$', fontsize=78)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, \ensuremath{\mathbf{a}}_j, d_{ij})$', fontsize=78)
        plt.ylim(config.plotting.ylim)
        plt.xlim([0, max_radius])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/true_func_{config_file}.tif", dpi=170.7)
        plt.close()

        true_func_list = []
        for k in trange(edges.shape[1]):
            n = type_list[to_numpy(edges[0, k])].astype(int)
            m = type_list[to_numpy(edges[1, k])].astype(int)
            true_func_list.append(true_func[3 * n.squeeze() + m.squeeze()])
        func_list = torch.stack(func_list) * ynorm
        true_func_list = torch.stack(true_func_list)
        rmserr_list = torch.sqrt(torch.mean((func_list - true_func_list) ** 2, axis=1))
        rmserr_list = to_numpy(rmserr_list)
        print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
        logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))


def plot_attraction_repulsion_continuous(config_file, epoch_list, log_dir, logger, device):

    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    dimension = config.simulation.dimension
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    dataset_name = config.dataset
    max_radius = config.simulation.max_radius
    n_runs = config.training.n_runs
    cmap = CustomColorMap(config=config)

    embedding_cluster = EmbeddingCluster(config)

    x_list, y_list, vnorm, ynorm = load_training_data(dataset_name, n_runs, log_dir, device)
    logger.info("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))
    model, bc_pos, bc_dpos = choose_training_model(config, device)

    x = x_list[1][0].clone().detach()

    for epoch in epoch_list:

        net = f"./log/try_{config_file}/models/best_model_with_1_graphs_{epoch}.pt"
        print(f'network: {net}')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])

        n_particle_types = 3
        index_particles = []
        for n in range(n_particle_types):
            index_particles.append(
                np.arange((n_particles // n_particle_types) * n, (n_particles // n_particle_types) * (n + 1)))

        fig, ax = fig_init()
        embedding = get_embedding(model.a, 1)
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=400, alpha=0.1)
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=78)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/first_embedding_{config_file}_{epoch}.tif", dpi=170.7)
        plt.close()

        fig, ax = fig_init()
        rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
        func_list = []
        for n in range(n_particles):
            embedding = model.a[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding), dim=1)
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            func = func[:, 0]
            func_list.append(func)
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color(n // 1600), linewidth=2, alpha=0.1)
        plt.xlabel(r'$d_{ij}$', fontsize=78)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=78)
        plt.xlim([0, max_radius])
        plt.ylim(config.plotting.ylim)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/func_{config_file}_{epoch}.tif", dpi=170.7)
        plt.close()

        fig, ax = fig_init()
        p = torch.load(f'graphs_data/graphs_{dataset_name}/model_p.pt')
        true_func_list = []
        csv_ = []
        csv_.append(to_numpy(rr))
        for n in range(n_particles):
            plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n // 1600), linewidth=2,
                     alpha=0.1)
            true_func_list.append(model.psi(rr, p[n], p[n]))
            csv_.append(to_numpy(model.psi(rr, p[n], p[n]).squeeze()))
        plt.xlabel(r'$d_{ij}$', fontsize=78)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=78)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlim([0, max_radius])
        plt.ylim(config.plotting.ylim)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/true_func_{config_file}.tif", dpi=170.7)
        np.save(f"./{log_dir}/results/true_func_{config_file}_{epoch}.npy", csv_)
        np.savetxt(f"./{log_dir}/results/true_func_{config_file}_{epoch}.txt", csv_)
        plt.close()

        func_list = torch.stack(func_list) * ynorm
        true_func_list = torch.stack(true_func_list)

        rmserr_list = torch.sqrt(torch.mean((func_list - true_func_list) ** 2, axis=1))
        rmserr_list = to_numpy(rmserr_list)
        print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
        logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))


def plot_gravity(config_file, epoch_list, log_dir, logger, device):

    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_particle_types = config.simulation.n_particle_types
    n_runs = config.training.n_runs
    cmap = CustomColorMap(config=config)
    dimension = config.simulation.dimension

    embedding_cluster = EmbeddingCluster(config)

    x_list, y_list, vnorm, ynorm = load_training_data(dataset_name, n_runs, log_dir, device)
    logger.info("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))
    x = x_list[1][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    n_particles = x.shape[0]

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    for epoch in epoch_list:

        net = f"./log/try_{config_file}/models/best_model_with_1_graphs_{epoch}.pt"
        print(f'network: {net}')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()

        model_a_first = model.a.clone().detach()

        fig,ax = fig_init()
        embedding = get_embedding(model.a, 1)
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], color=cmap.color(n), s=100, alpha=0.1)

        config.training.cluster_method = 'distance_embedding'
        config.training.cluster_distance_threshold = 0.01
        alpha=0.1
        accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config, config_file, embedding_cluster,
                                                                       cmap, index_particles, type_list,
                                                                       n_particle_types, n_particles, ynorm, epoch,
                                                                       log_dir, alpha, device)
        print(
            f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
        logger.info(
            f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()
        config.training.cluster_method = 'distance_plot'
        config.training.cluster_distance_threshold = 0.01
        alpha = 0.5
        accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config, config_file, embedding_cluster,
                                                                       cmap, index_particles, type_list,
                                                                       n_particle_types, n_particles, ynorm, epoch,
                                                                       log_dir, alpha, device)
        print(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
        logger.info(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')

        fig, ax = fig_init(formatx='%.3f', formaty='%.0f')
        p = torch.load(f'graphs_data/graphs_{dataset_name}/model_p.pt', map_location=device)
        rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
        rmserr_list = []
        for n in range(int(n_particles * (1 - config.training.particle_dropout))):
            embedding_ = model_a_first[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            func = func[:, 0]
            true_func = model.psi(rr, p[to_numpy(type_list[n]).astype(int)].squeeze(),
                                  p[to_numpy(type_list[n]).astype(int)].squeeze())
            rmserr_list.append(torch.sqrt(torch.mean((func * ynorm - true_func.squeeze()) ** 2)))
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=8, alpha=0.1)
        plt.xlabel(r'$d_{ij}$', fontsize=78)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=78)
        plt.xlim([0, 0.02])
        plt.ylim([0, 0.5E6])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/func_all_{config_file}_{epoch}.tif", dpi=170.7)
        rmserr_list = torch.stack(rmserr_list)
        rmserr_list = to_numpy(rmserr_list)
        print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
        logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
        plt.close()

        fig, ax = fig_init(formatx='%.3f', formaty='%.0f')
        plots = []
        plots.append(rr)
        for n in range(n_particle_types):
            plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n), linewidth=8)
            plots.append(model.psi(rr, p[n], p[n]).squeeze())
        plt.xlim([0, 0.02])
        plt.ylim([0, 0.5E6])
        plt.xlabel(r'$d_{ij}$', fontsize=78)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/true_func_{config_file}.tif", dpi=170.7)
        plt.close()

        rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
        plot_list = []
        for n in range(int(n_particles)):
            embedding_ = model_a_first[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
            with torch.no_grad():
                pred = model.lin_edge(in_features.float())
            pred = pred[:, 0]
            plot_list.append(pred * ynorm)
        p = np.linspace(0.5, 5, n_particle_types)
        p_list = p[to_numpy(type_list).astype(int)]
        popt_list = []
        for n in range(int(n_particles)):
            popt, pcov = curve_fit(power_model, to_numpy(rr), to_numpy(plot_list[n]))
            popt_list.append(popt)
        popt_list=np.array(popt_list)

        x_data = p_list.squeeze()
        y_data = popt_list[:, 0]
        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)

        if epoch=='20':

            threshold = 0.4
            relative_error = np.abs(y_data - x_data) / x_data
            pos = np.argwhere(relative_error < threshold)
            pos_outliers = np.argwhere(relative_error > threshold)

            if len(pos)>0:
                x_data_ = x_data[pos[:, 0]]
                y_data_ = y_data[pos[:, 0]]
                lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)
                residuals = y_data_ - linear_model(x_data_, *lin_fit)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data_)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                print(f'R^2$: {np.round(r_squared, 2)}  Slope: {np.round(lin_fit[0], 2)}  outliers: {np.sum(relative_error > threshold)}  ')
                logger.info(f'R^2$: {np.round(r_squared, 2)}  Slope: {np.round(lin_fit[0], 2)}  outliers: {np.sum(relative_error > threshold)}  ')

                fig, ax = fig_init()
                csv_ = []
                csv_.append(p_list)
                csv_.append(popt_list[:, 0])
                plt.plot(p_list, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
                plt.scatter(p_list, popt_list[:, 0], color='k', s=50, alpha=0.5)
                plt.scatter(p_list[pos_outliers[:, 0]], popt_list[pos_outliers[:, 0], 0], color='r', s=50)
                plt.xlabel(r'True mass ', fontsize=64)
                plt.ylabel(r'Learned mass ', fontsize=64)
                plt.xlim([0, 5.5])
                plt.ylim([0, 5.5])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/mass_{config_file}.tif", dpi=170)
                # csv_ = np.array(csv_)
                # np.save(f"./{log_dir}/results/mass_{config_file}.npy", csv_)
                # np.savetxt(f"./{log_dir}/results/mass_{config_file}.txt", csv_)
                plt.close()

                relative_error = np.abs(popt_list[:, 0] - p_list.squeeze()) / p_list.squeeze() * 100

                print(f'mass relative error: {np.round(np.mean(relative_error), 2)}+/-{np.round(np.std(relative_error), 2)}')
                print(f'mass relative error wo outliers: {np.round(np.mean(relative_error[pos[:, 0]]), 2)}+/-{np.round(np.std(relative_error[pos[:, 0]]), 2)}')
                logger.info(f'mass relative error: {np.round(np.mean(relative_error), 2)}+/-{np.round(np.std(relative_error), 2)}')
                logger.info(f'mass relative error wo outliers: {np.round(np.mean(relative_error[pos[:, 0]]), 2)}+/-{np.round(np.std(relative_error[pos[:, 0]]), 2)}')


                fig, ax = fig_init()
                csv_ = []
                csv_.append(p_list.squeeze())
                csv_.append(-popt_list[:, 1])
                csv_ = np.array(csv_)
                plt.plot(p_list, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
                plt.scatter(p_list, -popt_list[:, 1], color='k', s=50, alpha=0.5)
                plt.xlim([0, 5.5])
                plt.ylim([-4, 0])
                plt.xlabel(r'True mass', fontsize=78)
                plt.ylabel(r'Learned exponent', fontsize=78)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/exponent_{config_file}.tif", dpi=170)
                np.save(f"./{log_dir}/results/exponent_{config_file}.npy", csv_)
                np.savetxt(f"./{log_dir}/results/exponent_{config_file}.txt", csv_)
                plt.close()

                print(f'exponent: {np.round(np.mean(-popt_list[:, 1]), 2)}+/-{np.round(np.std(-popt_list[:, 1]), 2)}')
                logger.info(f'mass relative error: {np.round(np.mean(-popt_list[:, 1]), 2)}+/-{np.round(np.std(-popt_list[:, 1]), 2)}')

            else:
                print('no fit')
                logger.info('no fit')

            if os.path.exists(f"./{log_dir}/results/coeff_pysrr.npy"):
                popt_list = np.load(f"./{log_dir}/results/coeff_pysrr.npy")

            else:
                text_trap = StringIO()
                sys.stdout = text_trap
                popt_list = []
                for n in range(0,int(n_particles)):
                    model_pysrr, max_index, max_value = symbolic_regression(rr, plot_list[n])
                    # print(f'{p_list[n].squeeze()}/x0**2, {model_pysrr.sympy(max_index)}')
                    logger.info(f'{np.round(p_list[n].squeeze(),2)}/x0**2, pysrr found {model_pysrr.sympy(max_index)}')

                    expr = model_pysrr.sympy(max_index).as_terms()[0]
                    popt_list.append(expr[0][1][0][0])

                np.save(f"./{log_dir}/results/coeff_pysrr.npy", popt_list)

                # model_pysrr = PySRRegressor(
                #     niterations=30,  # < Increase me for better results
                #     random_state=0,
                #     temp_equation_file=False
                # )
                # model_pysrr.fit(to_numpy(rr[:, None]), to_numpy(plot_list[0]))

                sys.stdout = sys.__stdout__

                popt_list = np.array(popt_list)

            x_data = p_list.squeeze()
            y_data = popt_list
            lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)

            threshold = 0.4
            relative_error = np.abs(y_data - x_data) / x_data
            pos = np.argwhere(relative_error < threshold)
            x_data_ = x_data[pos[:, 0]]
            y_data_ = y_data[pos[:, 0]]
            lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)


            residuals = y_data_ - linear_model(x_data_, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data_ - np.mean(y_data_)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)


            print(f'R^2$: {np.round(r_squared, 2)}  Slope: {np.round(lin_fit[0], 2)}  outliers: {np.sum(relative_error > threshold)}  threshold: {threshold} ')
            logger.info(f'R^2$: {np.round(r_squared, 2)}  Slope: {np.round(lin_fit[0], 2)}  outliers: {np.sum(relative_error > threshold)}  threshold: {threshold} ')

            fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
            plt.scatter(x_data_,y_data_, color='k', s=1, alpha=0.5)
            plt.plot(p_list, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
            plt.scatter(p_list, popt_list, color='k', s=50, alpha=0.5)
            plt.xlabel(r'True mass ', fontsize=64)
            plt.ylabel(r'Learned mass ', fontsize=64)
            plt.xlim([0, 5.5])
            plt.ylim([0, 5.5])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/pysrr_mass_{config_file}.tif", dpi=300)
            plt.close()

            relative_error = np.abs(popt_list - p_list.squeeze()) / p_list.squeeze() * 100

            print(f'pysrr_mass relative error: {np.round(np.mean(relative_error), 2)}+/-{np.round(np.std(relative_error), 2)}')
            print(f'pysrr_mass relative error wo outliers: {np.round(np.mean(relative_error[pos[:, 0]]), 2)}+/-{np.round(np.std(relative_error[pos[:, 0]]), 2)}')
            logger.info(f'pysrr_mass relative error: {np.round(np.mean(relative_error), 2)}+/-{np.round(np.std(relative_error), 2)}')
            logger.info(f'pysrr_mass relative error wo outliers: {np.round(np.mean(relative_error[pos[:, 0]]), 2)}+/-{np.round(np.std(relative_error[pos[:, 0]]), 2)}')


def plot_gravity_continuous(config_file, epoch_list, log_dir, logger, device):

    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    n_runs = config.training.n_runs
    dimension= config.simulation.dimension
    cmap = CustomColorMap(config=config)

    embedding_cluster = EmbeddingCluster(config)

    time.sleep(0.5)

    x_list, y_list, vnorm, ynorm = load_training_data(dataset_name, n_runs, log_dir, device)
    logger.info("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))
    model, bc_pos, bc_dpos = choose_training_model(config, device)

    x = x_list[1][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)

    for epoch in epoch_list:

        net = f"./log/try_{config_file}/models/best_model_with_1_graphs_{epoch}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()

        embedding = get_embedding(model.a, 1)
        fig, ax = fig_init()
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], color=cmap.color(n % 256),
                        s=400,
                        alpha=0.5)
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=78)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/embedding_{config_file}.tif", dpi=170)
        plt.close()

        fig, ax = fig_init(formatx='%.3f', formaty='%.0f')
        p = torch.load(f'graphs_data/graphs_{dataset_name}/model_p.pt', map_location=device)
        rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
        rmserr_list = []
        csv_ = []
        csv_.append(to_numpy(rr))
        for n in range(int(n_particles * (1 - config.training.particle_dropout))):
            embedding_ = model.a[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            func = func[:, 0]
            csv_.append(to_numpy(func))
            true_func = model.psi(rr, p[to_numpy(type_list[n]).astype(int)].squeeze(),
                                  p[to_numpy(type_list[n]).astype(int)].squeeze())
            rmserr_list.append(torch.sqrt(torch.mean((func * ynorm - true_func.squeeze()) ** 2)))
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color(n % 256), linewidth=8, alpha=0.1)
        plt.xlabel(r'$d_{ij}$', fontsize=78)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=78)
        plt.xlim([0, max_radius])
        plt.xlim([0, 0.02])
        plt.ylim([0, 0.5E6])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/func_{config_file}.tif", dpi=170)
        csv_ = np.array(csv_)
        np.save(f"./{log_dir}/results/func_{config_file}.npy", csv_)
        np.savetxt(f"./{log_dir}/results/func_{config_file}.txt", csv_)
        plt.close()

        rmserr_list = torch.stack(rmserr_list)
        rmserr_list = to_numpy(rmserr_list)
        print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
        logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))

        fig, ax = fig_init(formatx='%.3f', formaty='%.0f')
        p = np.linspace(0.5, 5, n_particle_types)
        p = torch.tensor(p, device=device)
        csv_ = []
        csv_.append(to_numpy(rr))
        for n in range(n_particle_types - 1, -1, -1):
            plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n % 256), linewidth=8)
            csv_.append(to_numpy(model.psi(rr, p[n], p[n]).squeeze()))
        plt.xlim([0, 0.02])
        plt.ylim([0, 0.5E6])
        plt.xlabel(r'$d_{ij}$', fontsize=78)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/true_func_{config_file}.tif", dpi=300)
        csv_ = np.array(csv_)
        np.save(f"./{log_dir}/results/true_func_{config_file}.npy", csv_)
        np.savetxt(f"./{log_dir}/results/true_func_{config_file}.txt", csv_)
        plt.close()

        rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
        plot_list = []
        for n in range(int(n_particles * (1 - config.training.particle_dropout))):
            embedding_ = model.a[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
            with torch.no_grad():
                pred = model.lin_edge(in_features.float())
            pred = pred[:, 0]
            plot_list.append(pred * ynorm)
        p = np.linspace(0.5, 5, n_particle_types)
        p_list = p[to_numpy(type_list).astype(int)]
        popt_list = []
        for n in range(int(n_particles * (1 - config.training.particle_dropout))):
            popt, pcov = curve_fit(power_model, to_numpy(rr), to_numpy(plot_list[n]))
            popt_list.append(popt)
        popt_list=np.array(popt_list)

        x_data = p_list.squeeze()
        y_data = popt_list[:, 0]
        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)

        threshold = 0.4
        relative_error = np.abs(y_data - x_data) / x_data
        pos = np.argwhere(relative_error < threshold)
        pos_outliers = np.argwhere(relative_error > threshold)
        x_data_ = x_data[pos[:, 0]]
        y_data_ = y_data[pos[:, 0]]
        lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)
        residuals = y_data_ - linear_model(x_data_, *lin_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data_ - np.mean(y_data_)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f'R^2$: {np.round(r_squared, 2)}  Slope: {np.round(lin_fit[0], 2)}  outliers: {np.sum(relative_error > threshold)}  threshold: {threshold} ')
        logger.info(f'R^2$: {np.round(r_squared, 2)}  Slope: {np.round(lin_fit[0], 2)}  outliers: {np.sum(relative_error > threshold)}  threshold: {threshold} ')

        fig, ax = fig_init()
        plt.plot(p_list, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
        plt.scatter(p_list, popt_list[:, 0], color='k', s=50, alpha=0.5)
        plt.scatter(p_list[pos_outliers[:, 0]], popt_list[pos_outliers[:, 0], 0], color='r', s=50)
        plt.xlabel(r'True mass ', fontsize=64)
        plt.ylabel(r'Learned mass ', fontsize=64)
        plt.xlim([0, 5.5])
        plt.ylim([0, 5.5])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/mass_{config_file}.tif", dpi=300)
        plt.close()

        relative_error = np.abs(popt_list[:, 0] - p_list.squeeze()) / p_list.squeeze() * 100

        print(f'mass relative error: {np.round(np.mean(relative_error), 2)}+/-{np.round(np.std(relative_error), 2)}')
        print(f'mass relative error wo outliers: {np.round(np.mean(relative_error[pos[:, 0]]), 2)}+/-{np.round(np.std(relative_error[pos[:, 0]]), 2)}')
        logger.info(f'mass relative error: {np.round(np.mean(relative_error), 2)}+/-{np.round(np.std(relative_error), 2)}')
        logger.info(f'mass relative error wo outliers: {np.round(np.mean(relative_error[pos[:, 0]]), 2)}+/-{np.round(np.std(relative_error[pos[:, 0]]), 2)}')


        fig, ax = fig_init()
        csv_ = []
        csv_.append(p_list.squeeze())
        csv_.append(-popt_list[:, 1])
        csv_ = np.array(csv_)
        plt.plot(p_list, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
        plt.scatter(p_list, -popt_list[:, 1], color='k', s=50, alpha=0.5)
        plt.xlim([0, 5.5])
        plt.ylim([-4, 0])
        plt.xlabel(r'True mass', fontsize=78)
        plt.ylabel(r'Learned exponent', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/exponent_{config_file}.tif", dpi=300)
        np.save(f"./{log_dir}/results/exponent_{config_file}.npy", csv_)
        np.savetxt(f"./{log_dir}/results/exponent_{config_file}.txt", csv_)
        plt.close()

        print(f'exponent: {np.round(np.mean(-popt_list[:, 1]), 2)}+/-{np.round(np.std(-popt_list[:, 1]), 2)}')
        logger.info(f'mass relative error: {np.round(np.mean(-popt_list[:, 1]), 2)}+/-{np.round(np.std(-popt_list[:, 1]), 2)}')

        if os.path.exists(f"./{log_dir}/results/coeff_pysrr.npy"):
            popt_list = np.load(f"./{log_dir}/results/coeff_pysrr.npy")

        else:
            text_trap = StringIO()
            sys.stdout = text_trap
            popt_list = []
            for n in range(0,int(n_particles * (1 - config.training.particle_dropout))):
                print(n)
                model_pysrr, max_index, max_value = symbolic_regression(rr, plot_list[n])
                # print(f'{p_list[n].squeeze()}/x0**2, {model_pysrr.sympy(max_index)}')
                logger.info(f'{np.round(p_list[n].squeeze(),2)}/x0**2, pysrr found {model_pysrr.sympy(max_index)}')

                expr = model_pysrr.sympy(max_index).as_terms()[0]
                popt_list.append(expr[0][1][0][0])

            np.save(f"./{log_dir}/results/coeff_pysrr.npy", popt_list)

            sys.stdout = sys.__stdout__

            popt_list = np.array(popt_list)

        x_data = p_list.squeeze()
        y_data = popt_list
        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)

        threshold = 0.4
        relative_error = np.abs(y_data - x_data) / x_data
        pos = np.argwhere(relative_error < threshold)
        pos_outliers = np.argwhere(relative_error > threshold)
        x_data_ = x_data[pos[:, 0]]
        y_data_ = y_data[pos[:, 0]]
        lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)
        residuals = y_data_ - linear_model(x_data_, *lin_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data_)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f'R^2$: {np.round(r_squared, 2)}  Slope: {np.round(lin_fit[0], 2)}  outliers: {np.sum(relative_error > threshold)}  ')
        logger.info(f'R^2$: {np.round(r_squared, 2)}  Slope: {np.round(lin_fit[0], 2)}  outliers: {np.sum(relative_error > threshold)}  ')

        fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
        csv_ = []
        csv_.append(p_list)
        csv_.append(popt_list)
        plt.plot(p_list, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
        plt.scatter(p_list, popt_list, color='k', s=50, alpha=0.5)
        plt.xlabel(r'True mass ', fontsize=64)
        plt.ylabel(r'Learned mass ', fontsize=64)
        plt.xlim([0, 5.5])
        plt.ylim([0, 5.5])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/pysrr_mass_{config_file}.tif", dpi=300)
        # csv_ = np.array(csv_)
        # np.save(f"./{log_dir}/results/mass_{config_file}.npy", csv_)
        # np.savetxt(f"./{log_dir}/results/mass_{config_file}.txt", csv_)
        plt.close()

        relative_error = np.abs(popt_list - p_list.squeeze()) / p_list.squeeze() * 100

        print(f'pysrr_mass relative error: {np.round(np.mean(relative_error), 2)}+/-{np.round(np.std(relative_error), 2)}')
        print(f'pysrr_mass relative error wo outliers: {np.round(np.mean(relative_error[pos[:, 0]]), 2)}+/-{np.round(np.std(relative_error[pos[:, 0]]), 2)}')
        logger.info(f'pysrr_mass relative error: {np.round(np.mean(relative_error), 2)}+/-{np.round(np.std(relative_error), 2)}')
        logger.info(f'pysrr_mass relative error wo outliers: {np.round(np.mean(relative_error[pos[:, 0]]), 2)}+/-{np.round(np.std(relative_error[pos[:, 0]]), 2)}')


def plot_Coulomb(config_file, epoch_list, log_dir, logger, device):

    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    n_runs = config.training.n_runs
    cmap = CustomColorMap(config=config)
    dimension = config.simulation.dimension

    embedding_cluster = EmbeddingCluster(config)

    x_list, y_list, vnorm, ynorm = load_training_data(dataset_name, n_runs, log_dir, device)
    logger.info("vnorm:{:.2e},  ynorm:{:.2e}".format(to_numpy(vnorm), to_numpy(ynorm)))
    x = x_list[0][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    n_particles = x.shape[0]

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    for epoch in epoch_list:

        net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_{epoch}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()


        config.training.cluster_method = 'distance_plot'
        config.training.cluster_distance_threshold = 0.1
        alpha=0.5
        accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config, config_file, embedding_cluster,
                                                                       cmap, index_particles, type_list,
                                                                       n_particle_types, n_particles, ynorm, epoch,
                                                                       log_dir, alpha, device)
        print(
            f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
        logger.info(
            f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()
        config.training.cluster_method = 'distance_embedding'
        config.training.cluster_distance_threshold = 0.01
        alpha = 0.5
        accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config, config_file, embedding_cluster,
                                                                       cmap, index_particles, type_list,
                                                                       n_particle_types, n_particles, ynorm, epoch,
                                                                       log_dir, alpha, device)
        print(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
        logger.info(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')

        x = x_list[0][100].clone().detach()
        index_particles = get_index_particles(x, n_particle_types, dimension)
        type_list = to_numpy(get_type_list(x, dimension))
        distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
        adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
        edges = adj_t.nonzero().t().contiguous()
        indexes = np.random.randint(0, edges.shape[1], 5000)
        edges = edges[:, indexes]

        p = [2, 1, -1]

        fig, ax = fig_init(formatx='%.3f', formaty='%.0f')
        func_list = []
        rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
        table_qiqj = np.zeros((10,1))
        tmp = np.array([-2, -1, 1, 2, 4])
        table_qiqj[tmp.astype(int)+2]=np.arange(5)[:,None]
        qiqj_list=[]
        for n in trange(edges.shape[1]):
            embedding_1 = model.a[1, edges[0, n], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            embedding_2 = model.a[1, edges[1, n], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            qiqj = p[type_list[to_numpy(edges[0, n])].astype(int).squeeze()] * p[type_list[to_numpy(edges[1, n])].astype(int).squeeze()]
            qiqj_list.append(qiqj)
            type = table_qiqj[qiqj+2].astype(int).squeeze()
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding_1, embedding_2), dim=1)
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            func = func[:, 0]
            func_list.append(func * ynorm)
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color(type), linewidth=8, alpha=0.1)

        plt.xlabel(r'$d_{ij}$', fontsize=78)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, \ensuremath{\mathbf{a}}_j, d_{ij})$', fontsize=78)
        plt.xlim([0, 0.02])
        plt.ylim([-0.5E6, 0.5E6])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/func_{config_file}_{epoch}.tif", dpi=170.7)
        plt.close()

        fig, ax = fig_init(formatx='%.3f', formaty='%.0f')
        csv_ = []
        csv_.append(to_numpy(rr))
        true_func_list = []
        for n in trange(edges.shape[1]):
            temp = model.psi(rr, p[type_list[to_numpy(edges[0, n])].astype(int).squeeze()], p[type_list[to_numpy(edges[1, n])].astype(int).squeeze()] )
            true_func_list.append(temp)
            type = p[type_list[to_numpy(edges[0, n])].astype(int).squeeze()] * p[type_list[to_numpy(edges[1, n])].astype(int).squeeze()]
            type = table_qiqj[type+2].astype(int).squeeze()
            plt.plot(to_numpy(rr), np.array(temp.cpu()), linewidth=8, color=cmap.color(type))
            csv_.append(to_numpy(temp.squeeze()))
        plt.xlim([0, 0.02])
        plt.ylim([-0.5E6, 0.5E6])
        plt.xlabel(r'$d_{ij}$', fontsize=78)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, \ensuremath{\mathbf{a}}_j, d_{ij})$', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/true_func_{config_file}_{epoch}.tif", dpi=170.7)
        np.save(f"./{log_dir}/results/true_func_{config_file}_{epoch}.npy", csv_)
        np.savetxt(f"./{log_dir}/results/true_func_{config_file}_{epoch}.txt", csv_)
        plt.close()

        func_list = torch.stack(func_list)
        true_func_list = torch.stack(true_func_list)
        rmserr_list = torch.sqrt(torch.mean((func_list - true_func_list) ** 2, axis=1))
        rmserr_list = to_numpy(rmserr_list)
        print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
        logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))

        if os.path.exists(f"./{log_dir}/results/coeff_pysrr.npy"):
            popt_list = np.load(f"./{log_dir}/results/coeff_pysrr.npy")

        else:
            print('curve fitting ...')
            text_trap = StringIO()
            sys.stdout = text_trap
            popt_list = []
            qiqj_list = np.array(qiqj_list)
            for n in range(0,edges.shape[1],5):
                model_pysrr, max_index, max_value = symbolic_regression(rr, func_list[n])
                print(f'{-qiqj_list[n]}/x0**2, {model_pysrr.sympy(max_index)}')
                logger.info(f'{-qiqj_list[n]}/x0**2, pysrr found {model_pysrr.sympy(max_index)}')

                expr = model_pysrr.sympy(max_index).as_terms()[0]
                popt_list.append(-expr[0][1][0][0])

            np.save(f"./{log_dir}/results/coeff_pysrr.npy", popt_list)
            np.save(f"./{log_dir}/results/qiqj.npy", qiqj_list)

        qiqj_list = np.load(f"./{log_dir}/results/qiqj.npy")
        qiqj = []
        for n in range(0, len(qiqj_list), 5):
            qiqj.append(qiqj_list[n])
        qiqj_list = np.array(qiqj)

        threshold = 1

        fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
        x_data = qiqj_list.squeeze()
        y_data = popt_list.squeeze()
        lin_fit, r_squared, relative_error, not_outliers, x_data, y_data = linear_fit(x_data, y_data, threshold)
        plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
        plt.scatter(qiqj_list, popt_list, color='k', s=200, alpha=0.1)
        plt.xlim([-2.5, 5])
        plt.ylim([-2.5, 5])
        plt.ylabel(r'Learned $q_i q_j$', fontsize=64)
        plt.xlabel(r'True $q_i q_j$', fontsize=64)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/qiqj_{config_file}_{epoch}.tif", dpi=170)
        plt.close()

        print(f'slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}  outliers: {np.sum(relative_error > threshold)}   threshold: {threshold} ')
        logger.info(f'slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}  outliers: {np.sum(relative_error > threshold)}   threshold: {threshold} ')

        print(f'pysrr_qiqj relative error: {100*np.round(np.mean(relative_error), 2)}+/-{100*np.round(np.std(relative_error), 2)}')
        print(f'pysrr_qiqj relative error wo outliers: {100*np.round(np.mean(relative_error[not_outliers[:, 0]]), 2)}+/-{100*np.round(np.std(relative_error[not_outliers[:, 0]]), 2)}')
        logger.info(f'pysrr_qiqj relative error: {100*np.round(np.mean(relative_error), 2)}+/-{100*np.round(np.std(relative_error), 2)}')
        logger.info(f'pysrr_qiqj relative error wo outliers: {100*np.round(np.mean(relative_error[not_outliers[:, 0]]), 2)}+/-{100*np.round(np.std(relative_error[not_outliers[:, 0]]), 2)}')

        # qi retrieval


        qiqj = torch.tensor(popt_list, device=device)[:, None]
        qiqj = qiqj[not_outliers[:, 0]]

        model_qs = model_qiqj(3, device)
        optimizer = torch.optim.Adam(model_qs.parameters(), lr=1E-2)
        qiqj_list = []
        loss_list = []
        for it in trange(20000):

            sample = np.random.randint(0, qiqj.shape[0] - 10)
            qiqj_ = qiqj[sample:sample + 10]

            optimizer.zero_grad()
            qs = model_qs()
            distance = torch.sum((qiqj_[:, None] - qs[None, :]) ** 2, dim=2)
            result = distance.min(dim=1)
            min_value = result.values
            min_index = result.indices
            loss = torch.mean(min_value) + torch.max(min_value)
            loss.backward()
            optimizer.step()
            if it % 100 == 0:
                qiqj_list.append(to_numpy(model_qs.qiqj))
                loss_list.append(to_numpy(loss))
        qiqj_list = np.array(qiqj_list).squeeze()


        print('qi')
        print(np.round(to_numpy(model_qs.qiqj[2].squeeze()),3), np.round(to_numpy(model_qs.qiqj[1].squeeze()),3),np.round(to_numpy(model_qs.qiqj[0].squeeze()),3) )
        logger.info('qi')
        logger.info(f'{np.round(to_numpy(model_qs.qiqj[2].squeeze()),3)}, {np.round(to_numpy(model_qs.qiqj[1].squeeze()),3)}, {np.round(to_numpy(model_qs.qiqj[0].squeeze()),3)}' )

        fig, ax = fig_init()
        plt.plot(qiqj_list[:, 0], linewidth=4)
        plt.plot(qiqj_list[:, 1], linewidth=4)
        plt.plot(qiqj_list[:, 2], linewidth=4)
        plt.xlabel('iteration',fontsize=78)
        plt.ylabel(r'$q_i$',fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/qi_{config_file}_{epoch}.tif", dpi=170)


def plot_boids(config_file, epoch_list, log_dir, logger, device):

    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_particle_types = config.simulation.n_particle_types
    n_runs = config.training.n_runs
    has_cell_division = config.simulation.has_cell_division
    cmap = CustomColorMap(config=config)
    n_frames = config.simulation.n_frames
    dimension = config.simulation.dimension

    embedding_cluster = EmbeddingCluster(config)

    print('load data ...')
    x_list = []
    y_list = []

    x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_1.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_1.pt', map_location=device))
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device)
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)
    x = x_list[0][-1].clone().detach()

    print('done ...')

    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    n_particles = x.shape[0]
    if has_cell_division:
        T1_list = []
        T1_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/T1_list_1.pt', map_location=device))
        n_particles_max = np.load(os.path.join(log_dir, 'n_particles_max.npy'))
        config.simulation.n_particles_max = n_particles_max
        type_list = T1_list[0]
        n_particles = len(type_list)

    for epoch in epoch_list:

        model, bc_pos, bc_dpos = choose_training_model(config, device)
        model = Interaction_Particle_extract(config, device, aggr_type=config.graph_model.aggr_type, bc_dpos=bc_dpos)

        net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_{epoch}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()

        alpha = 0.5
        print('clustering ...')

        accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config, config_file, embedding_cluster,
                                                                       cmap, index_particles, type_list,
                                                                       n_particle_types, n_particles, ynorm, epoch,
                                                                       log_dir, alpha, device)
        print(
            f'final result     accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
        logger.info(
            f'final result     accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')

        if has_cell_division:
            plot_cell_rates(config, device, log_dir, n_particle_types, type_list, x_list, new_labels, cmap, logger)

        print('compare reconstructed interaction with ground truth...')

        p = torch.load(f'graphs_data/graphs_{dataset_name}/model_p.pt', map_location=device)
        model_B = PDE_B_extract(aggr_type=config.graph_model.aggr_type, p=torch.squeeze(p), bc_dpos=bc_dpos)

        fig, ax = fig_init()
        rr = torch.tensor(np.linspace(-max_radius, max_radius, 1000)).to(device)
        func_list = []
        true_func_list = []
        x = x_list[0][-1].clone().detach()
        for n in np.arange(len(x)):
            embedding_ = model.a[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            func = func[:, 0]
            type = to_numpy(x[n, 5]).astype(int)
            if type < n_particle_types:
                func_list.append(func)
                true_func = model_B.psi(rr, p[type])
                true_func_list.append(true_func)
                if (n % 10 == 0):
                    plt.plot(to_numpy(rr),
                             to_numpy(func) * to_numpy(ynorm),
                             color=cmap.color(type), linewidth=4, alpha=0.25)
        func_list = torch.stack(func_list)
        true_func_list = torch.stack(true_func_list)
        plt.ylim([-1E-4, 1E-4])
        plt.xlabel(r'$x_j-x_i$', fontsize=78)
        plt.ylabel(r'$f_{ij}$', fontsize=78)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        fmt = lambda x, pos: '{:.1f}e-5'.format((x) * 1e5, pos)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/func_dij_{config_file}_{epoch}.tif", dpi=300)
        plt.close()

        fig, ax = fig_init()
        for n in range(n_particle_types):
            true_func = model_B.psi(rr, p[n])
            plt.plot(to_numpy(rr), to_numpy(true_func), color=cmap.color(n), linewidth=4)
        plt.ylim([-1E-4, 1E-4])
        plt.xlabel(r'$x_j-x_i$', fontsize=78)
        plt.ylabel(r'$f_{ij}$', fontsize=78)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        fmt = lambda x, pos: '{:.1f}e-5'.format((x) * 1e5, pos)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/true_func_dij_{config_file}_{epoch}.tif", dpi=300)
        func_list = func_list * ynorm
        func_list_ = torch.clamp(func_list, min=torch.tensor(-1.0E-4, device=device),
                                 max=torch.tensor(1.0E-4, device=device))
        true_func_list_ = torch.clamp(true_func_list, min=torch.tensor(-1.0E-4, device=device),
                                      max=torch.tensor(1.0E-4, device=device))
        rmserr_list = torch.sqrt(torch.mean((func_list_ - true_func_list_) ** 2, 1))
        rmserr_list = to_numpy(rmserr_list)
        print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
        logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))

        if epoch=='20':

            lin_edge_out_list = []
            type_list = []
            diffx_list = []
            diffv_list = []
            cohesion_list=[]
            alignment_list=[]
            separation_list=[]
            r_list = []
            for it in range(0,n_frames//2,n_frames//40):
                x = x_list[0][it].clone().detach()
                particle_index = to_numpy(x[:, 0:1]).astype(int)
                x[:, 5:6] = torch.tensor(new_labels[particle_index],
                                         device=device)  # set label found by clustering and mapperd to ground truth
                pos = torch.argwhere(x[:, 5:6] < n_particle_types).squeeze()
                pos = to_numpy(pos[:, 0]).astype(int)  # filter out cluster not associated with ground truth
                x = x[pos, :]
                distance = torch.sum(bc_dpos(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, dim=2)  # threshold
                adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)) * 1.0
                edge_index = adj_t.nonzero().t().contiguous()
                dataset = data.Data(x=x, edge_index=edge_index)
                with torch.no_grad():
                    y, in_features, lin_edge_out = model(dataset, data_id=1, training=False, vnorm=vnorm,
                                                         phi=torch.zeros(1, device=device))  # acceleration estimation
                y = y * ynorm
                lin_edge_out = lin_edge_out * ynorm

                # compute ground truth output
                rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
                psi_output = []
                for n in range(n_particle_types):
                    with torch.no_grad():
                        psi_output.append(model.psi(rr, torch.squeeze(p[n])))
                        y_B, sum, cohesion, alignment, separation, diffx, diffv, r, type = model_B(dataset)  # acceleration estimation

                if it==0:
                    lin_edge_out_list=lin_edge_out
                    diffx_list=diffx
                    diffv_list=diffv
                    r_list=r
                    type_list=type
                    cohesion_list = cohesion
                    alignment_list = alignment
                    separation_list = separation
                else:
                    lin_edge_out_list=torch.cat((lin_edge_out_list,lin_edge_out),dim=0)
                    diffx_list=torch.cat((diffx_list,diffx),dim=0)
                    diffv_list=torch.cat((diffv_list,diffv),dim=0)
                    r_list=torch.cat((r_list,r),dim=0)
                    type_list=torch.cat((type_list,type),dim=0)
                    cohesion_list=torch.cat((cohesion_list,cohesion),dim=0)
                    alignment_list=torch.cat((alignment_list,alignment),dim=0)
                    separation_list=torch.cat((separation_list,separation),dim=0)

            type_list = to_numpy(type_list)

            print(f'fitting with known functions {len(type_list)} points ...')
            cohesion_fit = np.zeros(n_particle_types)
            alignment_fit = np.zeros(n_particle_types)
            separation_fit = np.zeros(n_particle_types)
            indexes = np.unique(type_list)
            indexes = indexes.astype(int)

            if False:
                for n in indexes:
                    pos = np.argwhere(type_list == n)
                    pos = pos[:, 0].astype(int)
                    xdiff = diffx_list[pos, 0:1]
                    vdiff = diffv_list[pos, 0:1]
                    rdiff = r_list[pos]
                    x_data = torch.concatenate((xdiff, vdiff, rdiff[:, None]), axis=1)
                    y_data = lin_edge_out_list[pos, 0:1]
                    xdiff = diffx_list[pos, 1:2]
                    vdiff = diffv_list[pos, 1:2]
                    rdiff = r_list[pos]
                    tmp = torch.concatenate((xdiff, vdiff, rdiff[:, None]), axis=1)
                    x_data = torch.cat((x_data, tmp), dim=0)
                    tmp = lin_edge_out_list[pos, 1:2]
                    y_data = torch.cat((y_data, tmp), dim=0)
                    model_pysrr, max_index, max_value = symbolic_regression_multi(x_data, y_data)

            for loop in range(2):
                for n in indexes:
                    pos = np.argwhere(type_list == n)
                    pos = pos[:, 0].astype(int)
                    xdiff = to_numpy(diffx_list[pos, :])
                    vdiff = to_numpy(diffv_list[pos, :])
                    rdiff = to_numpy(r_list[pos])
                    x_data = np.concatenate((xdiff, vdiff, rdiff[:, None]), axis=1)
                    y_data = to_numpy(torch.norm(lin_edge_out_list[pos, :], dim=1))
                    if loop == 0:
                        lin_fit, lin_fitv = curve_fit(boids_model, x_data, y_data, method='dogbox')
                    else:
                        lin_fit, lin_fitv = curve_fit(boids_model, x_data, y_data, method='dogbox', p0=p00)
                    cohesion_fit[int(n)] = lin_fit[0]
                    alignment_fit[int(n)] = lin_fit[1]
                    separation_fit[int(n)] = lin_fit[2]
                p00 = [np.mean(cohesion_fit[indexes]), np.mean(alignment_fit[indexes]), np.mean(separation_fit[indexes])]

            threshold = 0.25

            x_data = np.abs(to_numpy(p[:, 0]) * 0.5E-5)
            y_data = np.abs(cohesion_fit)
            x_data = x_data[indexes]
            y_data = y_data[indexes]
            lin_fit, r_squared, relative_error, not_outliers, x_data, y_data = linear_fit(x_data, y_data, threshold)

            fig, ax = fig_init()
            fmt = lambda x, pos: '{:.1f}e-4'.format((x) * 1e4, pos)
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
            for id, n in enumerate(indexes):
                plt.scatter(x_data[id], y_data[id], color=cmap.color(n), s=400)
            plt.xlabel(r'True cohesion coeff. ', fontsize=56)
            plt.ylabel(r'Fitted cohesion coeff. ', fontsize=56)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/cohesion_{config_file}_{epoch}.tif", dpi=300)
            plt.close()
            print()
            print(f'cohesion slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}  outliers: {np.sum(relative_error > threshold)}   threshold {threshold} ')
            logger.info(f'cohesion slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}  outliers: {np.sum(relative_error > threshold)}   threshold {threshold} ')

            x_data = np.abs(to_numpy(p[:, 1]) * 5E-4)
            y_data = alignment_fit
            x_data = x_data[indexes]
            y_data = y_data[indexes]
            lin_fit, r_squared, relative_error, not_outliers, x_data, y_data = linear_fit(x_data, y_data, threshold)

            fig, ax = fig_init()
            fmt = lambda x, pos: '{:.1f}e-2'.format((x) * 1e2, pos)
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
            for id, n in enumerate(indexes):
                plt.scatter(x_data[id], y_data[id], color=cmap.color(n), s=400)
            plt.xlabel(r'True alignement coeff. ', fontsize=56)
            plt.ylabel(r'Fitted alignement coeff. ', fontsize=56)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/alignment_{config_file}_{epoch}.tif", dpi=300)
            plt.close()
            print(f'alignment   slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}  outliers: {np.sum(relative_error > threshold)}  threshold {threshold} ')
            logger.info(f'alignment   slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}  outliers: {np.sum(relative_error > threshold)}  threshold {threshold} ')

            x_data = np.abs(to_numpy(p[:, 2]) * 1E-8)
            y_data = separation_fit
            x_data = x_data[indexes]
            y_data = y_data[indexes]
            lin_fit, r_squared, relative_error, not_outliers, x_data, y_data = linear_fit(x_data, y_data, threshold)

            fig, ax = fig_init()
            fmt = lambda x, pos: '{:.1f}e-7'.format((x) * 1e7, pos)
            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
            for id, n in enumerate(indexes):
                plt.scatter(x_data[id], y_data[id], color=cmap.color(n), s=400)
            plt.xlabel(r'True separation coeff. ', fontsize=56)
            plt.ylabel(r'Fitted separation coeff. ', fontsize=56)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/separation_{config_file}_{epoch}.tif", dpi=300)
            plt.close()
            print(f'separation   slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}  outliers: {np.sum(relative_error > threshold)}  threshold {threshold} ')
            logger.info(f'separation   slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}  outliers: {np.sum(relative_error > threshold)}  threshold {threshold} ')


def plot_wave(config_file, epoch_list, log_dir, logger, cc, device):
    # Load parameters from config file
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    n_nodes = config.simulation.n_nodes
    n_nodes_per_axis = int(np.sqrt(n_nodes))
    n_frames = config.simulation.n_frames
    n_runs = config.training.n_runs

    hnorm = torch.load(f'./log/try_{config_file}/hnorm.pt', map_location=device).to(device)

    x_mesh_list = []
    y_mesh_list = []
    time.sleep(0.5)
    for run in trange(n_runs):
        x_mesh = torch.load(f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
        x_mesh_list.append(x_mesh)
        h = torch.load(f'graphs_data/graphs_{dataset_name}/y_mesh_list_{run}.pt', map_location=device)
        y_mesh_list.append(h)
    h = y_mesh_list[0][0].clone().detach()

    print(f'hnorm: {to_numpy(hnorm)}')
    time.sleep(0.5)
    mesh_data = torch.load(f'graphs_data/graphs_{dataset_name}/mesh_data_1.pt', map_location=device)
    mask_mesh = mesh_data['mask']

    x_mesh = x_mesh_list[0][n_frames - 1].clone().detach()
    n_nodes = x_mesh.shape[0]
    print(f'N nodes: {n_nodes}')
    x_mesh = x_mesh_list[1][0].clone().detach()

    i0 = imread(f'graphs_data/{config.simulation.node_coeff_map}')
    coeff = i0[(to_numpy(x_mesh[:, 2]) * 255).astype(int), (to_numpy(x_mesh[:, 1]) * 255).astype(int)] / 255
    coeff = np.reshape(coeff, (n_nodes_per_axis, n_nodes_per_axis))
    vm = np.max(coeff)
    fig, ax = fig_init()
    fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    plt.imshow(coeff, cmap='grey', vmin=0, vmax=vm)
    plt.xlabel(r'$x$', fontsize=78)
    plt.ylabel(r'$y$', fontsize=78)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/true_wave_coeff_{config_file}.tif", dpi=300)
    plt.close
    fig, ax = fig_init()
    fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    plt.imshow(coeff, cmap='grey', vmin=0, vmax=vm)
    plt.xlabel(r'$x$', fontsize=78)
    plt.ylabel(r'$y$', fontsize=78)
    cbar = plt.colorbar(shrink=0.5)
    cbar.ax.tick_params(labelsize=32)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/true_wave_coeff_{config_file}_cbar.tif", dpi=300)
    plt.close

    for epoch in epoch_list:

        net = f"./log/try_{config_file}/models/best_model_with_1_graphs_{epoch}.pt"
        print(f'network: {net}')

        mesh_model_gene = choose_mesh_model(config=config, X1_mesh=x_mesh[:,1:3], device=device)

        mesh_model, bc_pos, bc_dpos = choose_training_model(config, device)
        state_dict = torch.load(net, map_location=device)
        mesh_model.load_state_dict(state_dict['model_state_dict'])
        mesh_model.eval()

        x_mesh = x_mesh_list[1][7000].clone().detach()
        mesh_data = torch.load(f'graphs_data/graphs_{dataset_name}/mesh_data_1.pt', map_location=device)
        dataset_mesh = data.Data(x=x_mesh, edge_index=mesh_data['edge_index'],
                                 edge_attr=mesh_data['edge_weight'], device=device)
        with torch.no_grad():
            pred_gene = mesh_model_gene(dataset_mesh)
            pred = mesh_model(dataset_mesh, data_id=1)

        fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
        plt.scatter(to_numpy(pred_gene),to_numpy(pred)*to_numpy(hnorm),s=1,c='k',alpha=0.1)
        plt.close()
        fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
        plt.scatter(to_numpy(mesh_model_gene.laplacian_u),to_numpy(mesh_model.laplacian_u),s=1,alpha=0.1)
        plt.close()

        fig, ax = fig_init(formatx='%.0f', formaty='%.1f')
        rr = torch.tensor(np.linspace(-1800, 1800, 200)).to(device)
        coeff = np.reshape(to_numpy(mesh_model_gene.coeff)/vm/255, (n_nodes_per_axis * n_nodes_per_axis))
        coeff = np.clip(coeff, a_min=0, a_max=1)
        popt_list = []
        func_list = []
        for n in trange(n_nodes):
            embedding_ = mesh_model.a[1, n, :] * torch.ones((200, 2), device=device)
            in_features = torch.cat((rr[:, None], embedding_), dim=1)
            with torch.no_grad():
                h = mesh_model.lin_phi(in_features.float()) * hnorm
            h = h[:, 0]
            popt, pcov = curve_fit(linear_model, to_numpy(rr.squeeze()), to_numpy(h.squeeze()))
            popt_list.append(popt)
            func_list.append(h)
            # plt.scatter(to_numpy(rr), to_numpy(h), c=f'{coeff[n]}', edgecolors='none',alpha=0.1)
            plt.scatter(to_numpy(rr), to_numpy(h), c='k',alpha=0.1)
        plt.xlabel(r'$\nabla^2 u_i$', fontsize=78)
        plt.ylabel(r'$\Phi(\ensuremath{\mathbf{a}}_{i},\nabla^2 u_i)$', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/functions_{config_file}_{epoch}.tif", dpi=300)
        plt.close()

        func_list = torch.stack(func_list)
        popt_list = np.array(popt_list)

        threshold=-1
        x_data = np.reshape(to_numpy(mesh_model_gene.coeff)/100, (n_nodes_per_axis*n_nodes_per_axis))
        y_data = popt_list[:, 0]
        # discard borders
        pos = np.argwhere(to_numpy(mask_mesh) == 1)
        x_data = x_data[pos[:, 0]]
        y_data = y_data[pos[:, 0]]

        lin_fit, r_squared, relative_error, not_outliers, x_data, y_data = linear_fit(x_data, y_data, threshold)
        print(
            f'slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}  N points {len(x_data)} ')
        logger.info(
            f'slope: {np.round(lin_fit[0], 2)}  R^2$: {np.round(r_squared, 3)}   N points {len(x_data)} ')

        fig, ax = fig_init(formatx='%.5f', formaty='%.5f')
        plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
        plt.scatter(x_data, y_data, s=200, c='k', alpha=0.1)
        plt.xlabel('True wave coeff.', fontsize=78)
        plt.ylabel('Learned wave coeff.', fontsize=78)
        fmt = lambda x, pos: '{:.1f}e-3'.format((x) * 1e3, pos)
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/scatter_coeff_{config_file}_{epoch}.tif", dpi=300)
        plt.close()

        t = np.array(popt_list)
        t = t[:, 0]
        t = np.reshape(t, (n_nodes_per_axis, n_nodes_per_axis))
        t = np.flipud(t)
        fig, ax = fig_init()
        fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        plt.imshow(t, cmap='grey')
        plt.xlabel(r'$x$', fontsize=78)
        plt.ylabel(r'$y$', fontsize=78)
        fmt = lambda x, pos: '{:.3%}'.format(x)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/wave_coeff_{config_file}_{epoch}.tif", dpi=300)
        plt.close()

        embedding = get_embedding(mesh_model.a, 1)
        fig, ax = fig_init()
        # plt.scatter(embedding[pos[:,0], 0], embedding[pos[:,0], 1], c=x_data, s=100, alpha=1, cmap='grey')
        plt.scatter(embedding[pos[:,0], 0], embedding[pos[:,0], 1], c='k', s=100, alpha=0.1)
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=78)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=78)
        # plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=78)
        # plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/embedding_{config_file}_{epoch}.tif", dpi=300)
        plt.close()


def plot_particle_field(config_file, epoch_list, log_dir, logger, cc, device):

    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    n_nodes = config.simulation.n_nodes
    n_node_types = config.simulation.n_node_types
    node_value_map = config.simulation.node_value_map
    has_video = 'video' in node_value_map
    n_nodes_per_axis = int(np.sqrt(n_nodes))
    n_frames = config.simulation.n_frames
    has_siren = 'siren' in config.graph_model.field_type
    has_siren_time = 'siren_with_time' in config.graph_model.field_type
    target_batch_size = config.training.batch_size
    has_ghost = config.training.n_ghosts > 0
    if config.training.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)  # create colormap for given config.graph_model
    embedding_cluster = EmbeddingCluster(config)
    n_runs = config.training.n_runs

    x_list = []
    y_list = []
    x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_1.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_1.pt', map_location=device))
    ynorm = torch.load(f'./log/try_{dataset_name}/ynorm.pt', map_location=device).to(device)
    vnorm = torch.load(f'./log/try_{dataset_name}/vnorm.pt', map_location=device).to(device)

    x_mesh_list = []
    y_mesh_list = []
    x_mesh = torch.load(f'graphs_data/graphs_{dataset_name}/x_mesh_list_0.pt', map_location=device)
    x_mesh_list.append(x_mesh)
    y_mesh = torch.load(f'graphs_data/graphs_{dataset_name}/y_mesh_list_0.pt', map_location=device)
    y_mesh_list.append(y_mesh)
    hnorm = torch.load(f'./log/try_{dataset_name}/hnorm.pt', map_location=device).to(device)

    mesh_data = torch.load(f'graphs_data/graphs_{dataset_name}/mesh_data_0.pt', map_location=device)
    mask_mesh = mesh_data['mask']
    mask_mesh = mask_mesh.repeat(batch_size, 1)

    # matplotlib.use("Qt5Agg")
    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})

    x_mesh = x_mesh_list[0][0].clone().detach()
    i0 = imread(f'graphs_data/{node_value_map}')
    if has_video:
        i0 = i0[0]
        target = i0[(to_numpy(x_mesh[:, 2]) * 100).astype(int), (to_numpy(x_mesh[:, 1]) * 100).astype(int)]
        target = np.reshape(target, (n_nodes_per_axis, n_nodes_per_axis))
    else:
        target = i0[(to_numpy(x_mesh[:, 2]) * 255).astype(int), (to_numpy(x_mesh[:, 1]) * 255).astype(int)] * 5000/255
        target = np.reshape(target, (n_nodes_per_axis, n_nodes_per_axis))
        # target = np.flipud(target)
    vm = np.max(target)
    if vm == 0:
        vm = 0.01

    fig, ax = fig_init()
    plt.imshow(target, cmap=cc, vmin=0, vmax=vm)
    fmtx = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
    fmty = lambda x, pos: '{:.1f}'.format((100 - x) / 100, pos)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmty))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmtx))
    plt.xlabel(r'$x$', fontsize=78)
    plt.ylabel(r'$y$', fontsize=78)
    # cbar = plt.colorbar(shrink=0.5)
    # cbar.ax.tick_params(labelsize=32)
    # cbar.set_label(r'$Coupling$',fontsize=78)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/target_field.tif", dpi=300)
    plt.close()

    print('create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)

    x = x_list[0][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    if has_ghost:
        ghosts_particles = Ghost_Particles(config, n_particles, device)
        if config.training.ghost_method == 'MLP':
            optimizer_ghost_particles = torch.optim.Adam([ghosts_particles.data], lr=5E-4)
        else:
            optimizer_ghost_particles = torch.optim.Adam([ghosts_particles.ghost_pos], lr=1E-4)
        mask_ghost = np.concatenate((np.ones(n_particles), np.zeros(config.training.n_ghosts)))
        mask_ghost = np.tile(mask_ghost, batch_size)
        mask_ghost = np.argwhere(mask_ghost == 1)
        mask_ghost = mask_ghost[:, 0].astype(int)
    index_nodes = []
    x_mesh = x_mesh_list[0][0].clone().detach()
    for n in range(n_node_types):
        index = np.argwhere(x_mesh[:, 5].detach().cpu().numpy() == -n - 1)
        index_nodes.append(index.squeeze())

    if has_siren:

        image_width = int(np.sqrt(n_nodes))
        if has_siren_time:
            model_f = Siren_Network(image_width=image_width, in_features=3, out_features=1, hidden_features=128,
                                    hidden_layers=5, outermost_linear=True, device=device, first_omega_0=80,
                                    hidden_omega_0=80.)
        else:
            model_f = Siren_Network(image_width=image_width, in_features=2, out_features=1, hidden_features=64,
                                    hidden_layers=3, outermost_linear=True, device=device, first_omega_0=80,
                                    hidden_omega_0=80.)
        model_f.to(device=device)
        model_f.eval()

    for epoch in epoch_list:
        print(f'epoch: {epoch}')

        net = f"./log/try_{config_file}/models/best_model_with_1_graphs_{epoch}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])

        if has_siren:
            net = f'./log/try_{config_file}/models/best_model_f_with_1_graphs_{epoch}.pt'
            state_dict = torch.load(net, map_location=device)
            model_f.load_state_dict(state_dict['model_state_dict'])

        model_a_first = model.a.clone().detach()
        config.training.cluster_method = 'distance_plot'
        config.training.cluster_distance_threshold = 0.01
        alpha = 0.1
        accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config, config_file, embedding_cluster,
                                                                       cmap, index_particles, type_list,
                                                                       n_particle_types, n_particles, ynorm, epoch,
                                                                       log_dir, alpha, device)
        print(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
        logger.info(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')


        fig, ax = fig_init()
        p = torch.load(f'graphs_data/graphs_{dataset_name}/model_p.pt', map_location=device)
        rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
        rmserr_list = []
        for n in range(int(n_particles * (1 - config.training.particle_dropout))):
            embedding_ = model_a_first[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding_), dim=1)
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
                func = func[:, 0]
            true_func = model.psi(rr, p[to_numpy(type_list[n]).astype(int)].squeeze(),
                                  p[to_numpy(type_list[n]).astype(int)].squeeze())
            rmserr_list.append(torch.sqrt(torch.mean((func * ynorm - true_func.squeeze()) ** 2)))
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=2, alpha=0.1)
        plt.xlabel(r'$d_{ij}$', fontsize=78)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=78)
        plt.xlim([0, max_radius])
        plt.ylim(config.plotting.ylim)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/func_all_{config_file}_{epoch}.tif", dpi=170.7)
        rmserr_list = torch.stack(rmserr_list)
        rmserr_list = to_numpy(rmserr_list)
        print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
        logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
        plt.close()


        match config.graph_model.field_type:

            case 'siren_with_time' | 'siren':

                os.makedirs(f"./{log_dir}/results/rotation", exist_ok=True)
                os.makedirs(f"./{log_dir}/results/rotation/generated1", exist_ok=True)
                os.makedirs(f"./{log_dir}/results/rotation/generated2", exist_ok=True)
                os.makedirs(f"./{log_dir}/results/rotation/target", exist_ok=True)
                os.makedirs(f"./{log_dir}/results/rotation/field", exist_ok=True)
                s_p = 100

                x_mesh = x_mesh_list[0][0].clone().detach()
                i0 = imread(f'graphs_data/{node_value_map}')

                print('Output per frame ...')

                RMSE_list = []
                PSNR_list = []
                SSIM_list = []
                for frame in trange(0, n_frames):
                    x = x_list[0][frame].clone().detach()
                    fig, ax = fig_init(formatx='%.1f', formaty='%.1f')
                    plt.xlabel(r'$x$', fontsize=78)
                    plt.ylabel(r'$y$', fontsize=78)
                    for n in range(n_particle_types):
                        plt.scatter(to_numpy(x[index_particles[n], 2]), to_numpy(x[index_particles[n], 1]),
                                    s=s_p/2,
                                    color='k')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/video/generated1/generated_1_{epoch}_{frame}.tif",
                                dpi=150)
                    plt.close()

                    fig, ax = fig_init(formatx='%.1f', formaty='%.1f')
                    plt.xlabel(r'$x$', fontsize=78)
                    plt.ylabel(r'$y$', fontsize=78)
                    for n in range(n_particle_types):
                        plt.scatter(to_numpy(x[index_particles[n], 2]), to_numpy(x[index_particles[n], 1]),
                                    s=s_p)
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/video/generated2/generated_2_{epoch}_{frame}.tif",
                                dpi=150)
                    plt.close()

                    i0_ = i0[frame]
                    y = i0_[(to_numpy(x_mesh[:, 2]) * 100).astype(int), (to_numpy(x_mesh[:, 1]) * 100).astype(int)]
                    y = np.reshape(y, (n_nodes_per_axis, n_nodes_per_axis))
                    fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
                    plt.imshow(y, cmap=cc, vmin=0, vmax=vm)
                    plt.xlabel(r'$x$', fontsize=78)
                    plt.ylabel(r'$y$', fontsize=78)
                    fmtx = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
                    fmty = lambda x, pos: '{:.1f}'.format((100-x) / 100, pos)
                    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmty))
                    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmtx))
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/video/target/target_field_{epoch}_{frame}.tif",
                                dpi=150)
                    plt.close()

                    pred = model_f(time=frame / n_frames) ** 2
                    pred = torch.reshape(pred, (n_nodes_per_axis, n_nodes_per_axis))
                    pred = to_numpy(torch.sqrt(pred))
                    pred = np.flipud(pred)
                    fig, ax = fig_init(formatx='%.0f', formaty='%.0f')
                    pred = np.rot90(pred,1)
                    pred = np.fliplr(pred)
                    # pred = np.flipud(pred)
                    plt.imshow(pred, cmap=cc, vmin=0, vmax=vm)
                    plt.xlabel(r'$x$', fontsize=78)
                    plt.ylabel(r'$y$', fontsize=78)
                    fmtx = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
                    fmty = lambda x, pos: '{:.1f}'.format((100-x) / 100, pos)
                    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmty))
                    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmtx))
                    plt.tight_layout()
                    plt.savefig(f"./{log_dir}/results/video/field/reconstructed_field_{epoch}_{frame}.tif",
                                dpi=150)
                    plt.close()

                    RMSE = np.sqrt(np.mean((y - pred) ** 2))
                    RMSE_list = np.concatenate((RMSE_list, [RMSE]))
                    PSNR = calculate_psnr(y, pred, max_value=np.max(y))
                    PSNR_list = np.concatenate((PSNR_list, [PSNR]))
                    SSIM = calculate_ssim(y, pred)
                    SSIM_list = np.concatenate((SSIM_list, [SSIM]))
                    if frame==0:
                        y_list = [y]
                        pred_list = [pred]
                    else:
                        y_list = np.concatenate((y_list, [y]))
                        pred_list = np.concatenate((pred_list, [pred]))

                fig, ax = fig_init(formatx='%.2f', formaty='%.2f')
                plt.scatter(y_list, pred_list, color='k', s=0.1, alpha=0.01)
                plt.xlabel(r'True $b_i(t)$', fontsize=78)
                plt.ylabel(r'Recons. $b_i(t)$', fontsize=78)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/cues_scatter_{epoch}.tif", dpi=170)
                plt.close()

                r, p_value = pearsonr(y_list.flatten(), pred_list.flatten())
                print(f"Pearson's r: {r:.4f}, p-value: {p_value:.6f}")
                logger.info(f"Pearson's r: {r:.4f}, p-value: {p_value:.6f}")


                fig, ax = fig_init()
                plt.scatter(np.linspace(0, n_frames, len(SSIM_list)), SSIM_list, color='k', linewidth=4)
                plt.xlabel(r'$Frame$', fontsize=78)
                plt.ylabel(r'$SSIM$', fontsize=78)
                plt.ylim([0, 1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/ssim_{epoch}.tif", dpi=150)
                plt.close()

                print(f'SSIM: {np.round(np.mean(SSIM_list), 3)}+/-{np.round(np.std(SSIM_list), 3)}')

                fig, ax = fig_init()
                plt.scatter(np.linspace(0, n_frames, len(SSIM_list)), RMSE_list, color='k', linewidth=4)
                plt.xlabel(r'$Frame$', fontsize=78)
                plt.ylabel(r'RMSE', fontsize=78)
                plt.ylim([0, 1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/rmse_{epoch}.tif", dpi=150)
                plt.close()

                fig, ax = fig_init()
                plt.scatter(np.linspace(0, n_frames, len(SSIM_list)), PSNR_list, color='k', linewidth=4)
                plt.xlabel(r'$Frame$', fontsize=78)
                plt.ylabel(r'PSNR', fontsize=78)
                plt.ylim([0, 50])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/psnr_{epoch}.tif", dpi=150)
                plt.close()

            case 'tensor':

                fig, ax = fig_init()
                pts = to_numpy(torch.reshape(model.field[1], (100, 100)))
                pts = np.flipud(pts)
                plt.imshow(pts, cmap=cc)
                fmtx = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
                fmty = lambda x, pos: '{:.1f}'.format((100 - x) / 100, pos)
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmty))
                ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmtx))
                plt.xlabel(r'$x$', fontsize=78)
                plt.ylabel(r'$y$', fontsize=78)
                # cbar = plt.colorbar(shrink=0.5)
                # cbar.ax.tick_params(labelsize=32)
                # cbar.set_label(r'$Coupling$',fontsize=78)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/field_{config_file}_{epoch}.tif", dpi=300)
                # np.save(f"./{log_dir}/results/embedding_{config_file}.npy", csv_)
                # csv_= np.reshape(csv_,(csv_.shape[0]*csv_.shape[1],2))
                # np.savetxt(f"./{log_dir}/results/embedding_{config_file}.txt", csv_)
                plt.close()
                rmse = np.sqrt(np.mean((target - pts) ** 2))
                print(f'RMSE: {rmse}')
                logger.info(f'RMSE: {rmse}')

                fig, ax = fig_init()
                plt.scatter(target, pts, c='k', s=10, alpha=0.1)
                plt.xlabel(r'True $b_i$', fontsize=78)
                plt.ylabel(r'Recons. $b_i$', fontsize=78)

                x_data = np.reshape(pts, (n_nodes))
                y_data = np.reshape(target, (n_nodes))
                threshold = 0.25
                relative_error = np.abs(y_data - x_data)
                print(f'outliers: {np.sum(relative_error > threshold)} / {n_particles}')
                pos = np.argwhere(relative_error < threshold)

                x_data_ = x_data[pos].squeeze()
                y_data_ = y_data[pos].squeeze()

                lin_fit, lin_fitv = curve_fit(linear_model, x_data_, y_data_)
                residuals = y_data_ - linear_model(x_data_, *lin_fit)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                plt.plot(x_data_, linear_model(x_data_, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
                plt.xlim([0, 2])
                plt.ylim([-0, 2])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/results/field_scatter_{config_file}_{epoch}.tif", dpi=300)

                print(f'R^2$: {np.round(r_squared, 3)}  Slope: {np.round(lin_fit[0], 2)}')
                logger.info(f'R^2$: {np.round(r_squared, 3)}  Slope: {np.round(lin_fit[0], 2)}')


def plot_RD_RPS(config_file, epoch_list, log_dir, logger, cc, device):
    # Load parameters from config file
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    n_nodes = config.simulation.n_nodes
    n_nodes_per_axis = int(np.sqrt(n_nodes))
    n_node_types = config.simulation.n_node_types
    n_frames = config.simulation.n_frames
    n_runs = config.training.n_runs
    delta_t = config.simulation.delta_t
    cmap = CustomColorMap(config=config)

    embedding_cluster = EmbeddingCluster(config)

    hnorm = torch.load(f'./log/try_{config_file}/hnorm.pt', map_location=device).to(device)

    x_mesh_list = []
    y_mesh_list = []
    time.sleep(0.5)
    for run in trange(n_runs):
        x_mesh = torch.load(f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt', map_location=device)
        x_mesh_list.append(x_mesh)
        h = torch.load(f'graphs_data/graphs_{dataset_name}/y_mesh_list_{run}.pt', map_location=device)
        y_mesh_list.append(h)
    h = y_mesh_list[0][0].clone().detach()

    print(f'hnorm: {to_numpy(hnorm)}')
    time.sleep(0.5)
    mesh_data = torch.load(f'graphs_data/graphs_{dataset_name}/mesh_data_1.pt', map_location=device)
    mask_mesh = mesh_data['mask']
    edge_index_mesh = mesh_data['edge_index']
    edge_weight_mesh = mesh_data['edge_weight']

    x_mesh = x_mesh_list[1][0].clone().detach()

    i0 = imread(f'graphs_data/{config.simulation.node_coeff_map}')
    coeff = i0[(to_numpy(x_mesh[:, 2]) * 255).astype(int), (to_numpy(x_mesh[:, 1]) * 255).astype(int)]
    coeff = np.reshape(coeff, (n_nodes_per_axis, n_nodes_per_axis))
    vm = np.max(coeff)
    print(f'vm: {vm}')

    fig, ax = fig_init()
    fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    plt.imshow(np.flipud(coeff), vmin=0, vmax=vm, cmap='grey')
    plt.xlabel(r'$x$', fontsize=78)
    plt.ylabel(r'$y$', fontsize=78)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/true_coeff_{config_file}.tif", dpi=300)
    plt.close()
    fig, ax = fig_init()
    fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    plt.imshow(coeff, vmin=0, vmax=vm, cmap='grey')
    plt.xlabel(r'$x$', fontsize=78)
    plt.ylabel(r'$y$', fontsize=78)
    cbar = plt.colorbar(shrink=0.5)
    cbar.ax.tick_params(labelsize=32)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/true_coeff_{config_file}_cbar.tif", dpi=300)
    plt.close()

    for epoch in epoch_list:

        net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_{epoch}.pt"

        model, bc_pos, bc_dpos = choose_training_model(config, device)
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        print(f'net: {net}')
        embedding = get_embedding(model.a, 1)

        cluster_method = 'distance_embedding'
        cluster_distance_threshold = 0.01
        labels, n_clusters = embedding_cluster.get(embedding, 'distance', thresh=cluster_distance_threshold)
        labels_map = np.reshape(labels, (n_nodes_per_axis, n_nodes_per_axis))
        fig, ax = fig_init()
        plt.imshow(labels_map, cmap='tab20', vmin=0, vmax=10)
        fmt = lambda x, pos: '{:.1f}'.format((x) / 100, pos)
        ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
        plt.xlabel(r'$x$', fontsize=78)
        plt.ylabel(r'$y$', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/labels_map_{config_file}_cbar.tif", dpi=300)
        plt.close

        fig, ax = fig_init()
        for nodes_type in np.unique(labels[labels <5]):
            pos = np.argwhere(labels == nodes_type)
            plt.scatter(embedding[pos, 0], embedding[pos, 1], s=400, cmap=cmap.color(nodes_type*2))
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=78)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/embedding_{config_file}_{epoch}.tif", dpi=300)
        plt.close()

        if True:

            k = 2400

            # collect data
            x_mesh = x_mesh_list[1][k].clone().detach()
            dataset = data.Data(x=x_mesh, edge_index=edge_index_mesh, edge_attr=edge_weight_mesh, device=device)
            with torch.no_grad():
                pred, laplacian_uvw, uvw, embedding, input_phi = model(dataset, data_id=1, return_all=True)
            pred = pred * hnorm
            y = y_mesh_list[1][k].clone().detach()

            # RD_RPS_model :
            c_ = torch.zeros(n_node_types, 1, device=device)
            for n in range(n_node_types):
                c_[n] = torch.tensor(config.simulation.diffusion_coefficients[n])
            c = c_[to_numpy(dataset.x[:, 5])].squeeze()
            c = torch.tensor(np.reshape(coeff,(n_nodes_per_axis*n_nodes_per_axis)),device=device)
            u = uvw[:, 0]
            v = uvw[:, 1]
            w = uvw[:, 2]
            # laplacian = mesh_model.beta * c * self.propagate(edge_index, x=(x, x), edge_attr=edge_attr)
            laplacian_u = c * laplacian_uvw[:, 0]
            laplacian_v = c * laplacian_uvw[:, 1]
            laplacian_w = c * laplacian_uvw[:, 2]
            a = 0.6
            p = u + v + w
            du = laplacian_u + u * (1 - p - a * v)
            dv = laplacian_v + v * (1 - p - a * w)
            dw = laplacian_w + w * (1 - p - a * u)
            increment = torch.cat((du[:, None], dv[:, None], dw[:, None]), dim=1)
            increment = increment.squeeze()

            lin_fit_true = np.zeros((len(np.unique(labels))-1, 3, 10))
            lin_fit_reconstructed = np.zeros((len(np.unique(labels))-1, 3, 10))
            eq_list = ['u', 'v', 'w']
            # class 0 is discarded (borders)
            for n in np.unique(labels)[1:]-1:
                print(n)
                pos = np.argwhere((labels == n+1) & (to_numpy(mask_mesh.squeeze()) == 1))
                pos = pos[:, 0].astype(int)
                for it, eq in enumerate(eq_list):
                    fitting_model = reaction_diffusion_model(eq)
                    laplacian_u = to_numpy(laplacian_uvw[pos, 0])
                    laplacian_v = to_numpy(laplacian_uvw[pos, 1])
                    laplacian_w = to_numpy(laplacian_uvw[pos, 2])
                    u = to_numpy(uvw[pos, 0])
                    v = to_numpy(uvw[pos, 1])
                    w = to_numpy(uvw[pos, 2])
                    x_data = np.concatenate((laplacian_u[:, None], laplacian_v[:, None], laplacian_w[:, None],
                                             u[:, None], v[:, None], w[:, None]), axis=1)
                    y_data = to_numpy(increment[pos, 0 + it:1 + it])
                    p0 = np.ones((10, 1))
                    lin_fit, lin_fitv = curve_fit(fitting_model, np.squeeze(x_data), np.squeeze(y_data),
                                                  p0=np.squeeze(p0), method='trf')
                    lin_fit_true[n, it] = lin_fit
                    y_data = to_numpy(pred[pos, it:it + 1])
                    lin_fit, lin_fitv = curve_fit(fitting_model, np.squeeze(x_data), np.squeeze(y_data),
                                                  p0=np.squeeze(p0), method='trf')
                    lin_fit_reconstructed[n, it] = lin_fit

            coeff_reconstructed = np.round(np.median(lin_fit_reconstructed, axis=0), 2)
            diffusion_coeff_reconstructed = np.round(np.median(lin_fit_reconstructed, axis=1), 2)[:, 9]
            coeff_true = np.round(np.median(lin_fit_true, axis=0), 2)
            diffusion_coeff_true = np.round(np.median(lin_fit_true, axis=1), 2)[:, 9]

            print(f'frame {k}')
            print(f'coeff_reconstructed: {coeff_reconstructed}')
            print(f'diffusion_coeff_reconstructed: {diffusion_coeff_reconstructed}')
            print(f'coeff_true: {coeff_true}')
            print(f'diffusion_coeff_true: {diffusion_coeff_true}')

            cp = ['uu', 'uv', 'uw', 'vv', 'vw', 'ww', 'u', 'v', 'w']
            results = {
                'True': coeff_true[0, 0:9],
                'Learned': coeff_reconstructed[0, 0:9],
            }
            x = np.arange(len(cp))  # the label locations
            width = 0.25  # the width of the bars
            multiplier = 0
            fig, ax = fig_init()
            for attribute, measurement in results.items():
                offset = width * multiplier
                rects = ax.bar(x + offset, measurement, width, label=attribute)
                multiplier += 1
            ax.set_ylabel('Polynomial coefficient', fontsize=78)
            ax.set_xticks(x + width, cp, fontsize=36)
            plt.title('First equation', fontsize=56)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/first_equation_{config_file}_{epoch}.tif", dpi=300)
            plt.close()
            cp = ['uu', 'uv', 'uw', 'vv', 'vw', 'ww', 'u', 'v', 'w']
            results = {
                'True': coeff_true[1, 0:9],
                'Learned': coeff_reconstructed[1, 0:9],
            }
            x = np.arange(len(cp))  # the label locations
            width = 0.25  # the width of the bars
            multiplier = 0
            fig, ax = fig_init()
            for attribute, measurement in results.items():
                offset = width * multiplier
                rects = ax.bar(x + offset, measurement, width, label=attribute)
                multiplier += 1
            ax.set_ylabel('Polynomial coefficient', fontsize=78)
            ax.set_xticks(x + width, cp, fontsize=36)
            plt.title('Second equation', fontsize=56)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/second_equation_{config_file}_{epoch}.tif", dpi=300)
            plt.close()
            cp = ['uu', 'uv', 'uw', 'vv', 'vw', 'ww', 'u', 'v', 'w']
            results = {
                'True': coeff_true[2, 0:9],
                'Learned': coeff_reconstructed[2, 0:9],
            }
            x = np.arange(len(cp))  # the label locations
            width = 0.25  # the width of the bars
            multiplier = 0
            fig, ax = fig_init()
            for attribute, measurement in results.items():
                offset = width * multiplier
                rects = ax.bar(x + offset, measurement, width, label=attribute)
                multiplier += 1
            ax.set_ylabel('Polynomial coefficient', fontsize=78)
            ax.set_xticks(x + width, cp, fontsize=36)
            plt.title('Third equation', fontsize=56)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/third_equation_{config_file}_{epoch}.tif", dpi=300)
            plt.close()

            true_diffusion_coeff = [0.01, 0.02, 0.03, 0.04]

            fig, ax = fig_init(formatx='%.3f', formaty='%.3f')
            x_data = np.array(true_diffusion_coeff)
            y_data = diffusion_coeff_reconstructed
            plt.scatter(x_data, y_data, c='k', s=400)
            plt.ylabel(r'Learned diffusion coeff.', fontsize=64)
            plt.xlabel(r'True diffusion coeff.', fontsize=64)
            plt.xlim([0, vm * 1.1])
            plt.ylim([0, vm * 1.1])
            lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
            residuals = y_data - linear_model(x_data, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/scatter_{config_file}_{epoch}.tif", dpi=300)
            plt.close()

            print(f"R^2$: {np.round(r_squared, 3)}  Slope: {np.round(lin_fit[0], 2)}")



            fig, ax = fig_init(formatx='%.3f', formaty='%.3f')
            x_data = coeff_true.flatten()
            y_data = coeff_reconstructed.flatten()
            plt.scatter(x_data, y_data, c='k', s=400)
            plt.ylabel(r'Learned coeff.', fontsize=64)
            plt.xlabel(r'True  coeff.', fontsize=64)
            lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
            residuals = y_data - linear_model(x_data, *lin_fit)
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
            plt.tight_layout()
            print(f"R^2$: {np.round(r_squared, 3)}  Slope: {np.round(lin_fit[0], 2)}")


def plot_signal(config_file, epoch_list, log_dir, logger, cc, device):
    # Load parameters from config file
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    n_frames = config.simulation.n_frames
    n_runs = config.training.n_runs
    n_particle_types = config.simulation.n_particle_types
    delta_t = config.simulation.delta_t
    cmap = CustomColorMap(config=config)
    dimension = config.simulation.dimension

    embedding_cluster = EmbeddingCluster(config)

    x_list = []
    y_list = []
    for run in trange(2):
        x = torch.load(f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt', map_location=device)
        y = torch.load(f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt', map_location=device)
        x_list.append(x)
        y_list.append(y)
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'))
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'))
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')

    print('Update variables ...')
    x = x_list[0][n_frames - 1].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)
    n_particles = x.shape[0]
    print(f'N particles: {n_particles}')
    config.simulation.n_particles = n_particles

    if 'mat' in config.simulation.connectivity_file:
        mat = scipy.io.loadmat(config.simulation.connectivity_file)
        adjacency = torch.tensor(mat['A'], device=device)
    else:
        adjacency = torch.load(config.simulation.connectivity_file, map_location=device)
    adj_t = adjacency > 0
    edge_index = adj_t.nonzero().t().contiguous()

    for epoch in epoch_list:

        net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_{epoch}.pt"
        model, bc_pos, bc_dpos = choose_training_model(config, device)
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.edges = edge_index
        print(f'net: {net}')

        config.training.cluster_method = 'distance_plot'
        config.training.cluster_distance_threshold = 0.01
        alpha = 0.1
        accuracy, n_clusters, new_labels = plot_embedding_func_cluster(model, config, config_file, embedding_cluster,
                                                                       cmap, index_particles, type_list,
                                                                       n_particle_types, n_particles, ynorm, epoch,
                                                                       log_dir, alpha, device)
        print(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
        logger.info(f'result accuracy: {np.round(accuracy, 2)}    n_clusters: {n_clusters}    obtained with  method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')

        fig, ax = fig_init()
        ax.xaxis.get_major_formatter()._usetex = False
        ax.yaxis.get_major_formatter()._usetex = False
        rr = torch.tensor(np.linspace(0, 4, 1000)).to(device)
        print(n_particles)
        func_list, proj_interaction = analyze_edge_function(rr=rr, vizualize=True, config=config,
                                                            model_MLP=model.lin_phi, model_a=model.a,
                                                            dataset_number=1,
                                                            n_particles=int(n_particles * (1 - config.training.particle_dropout)),
                                                            ynorm=ynorm,
                                                            type_list=to_numpy(x[:, 5]),
                                                            cmap=cmap, device=device)
        # plt.xlabel(r'$d_{ij}$', fontsize=78)
        # plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=78)
        plt.xlabel(r'$u$', fontsize=78)
        plt.ylabel(r'Learned $\Phi(u)$', fontsize=78)
        # plt.ylim([-0.05,0.15])
        plt.xlim([0,2])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/reconstructed_phi_u_{config_file}_{epoch}.tif", dpi=170.7)
        plt.close()

        fig, ax = fig_init()
        embedding_ = model.a[1, :, :]
        u = torch.tensor(0.5, device=device).float()
        u = u * torch.ones((n_particles, 1), device=device)
        in_features = torch.cat((u, embedding_), dim=1)
        with torch.no_grad():
            func = model.lin_phi(in_features.float())
        func = func[:, 0]
        proj_interaction = to_numpy(func[:, None])
        labels, n_clusters = embedding_cluster.get(proj_interaction, 'kmeans_auto')
        for n in range(n_clusters):
            pos = np.argwhere(labels == n)
            pos = np.array(pos)
            if pos.size > 0:
                plt.scatter(np.ones_like(pos) * 0.5, proj_interaction[pos, 0], color=cmap.color(n), s=400, alpha=0.1)
        label_list = []
        for n in range(n_particle_types):
            tmp = labels[index_particles[n]]
            label_list.append(np.round(np.median(tmp)))
        label_list = np.array(label_list)
        plt.xlabel(r'$u$', fontsize=78)
        plt.ylabel(r'$\Phi(u)$', fontsize=78)
        plt.ylim([-0.25, 0.25])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/cluster_{config_file}_{epoch}.tif", dpi=300)
        plt.close()

        new_labels = labels.copy()
        for n in range(n_particle_types):
            new_labels[labels == label_list[n]] = n
        accuracy = metrics.accuracy_score(to_numpy(type_list), new_labels)
        print(f'accuracy: {np.round(accuracy, 2)}   n_clusters: {n_clusters}')
        logger.info(f'accuracy: {np.round(accuracy, 2)}   n_clusters: {n_clusters}')

        fig, ax = fig_init()
        for n in np.unique(new_labels):
            pos = np.argwhere(new_labels == n)
            plt.scatter(to_numpy(x[pos, 2]), to_numpy(x[pos, 1]), s=200)
        plt.xlim([-1.2, 1.2])
        plt.ylim([-1.2, 1.2])
        plt.xlabel(r'$x$', fontsize=78)
        plt.ylabel(r'$y$', fontsize=78)
        plt.xticks(fontsize=48.0)
        plt.yticks(fontsize=48.0)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/classif_{config_file}_{epoch}.tif", dpi=300)
        plt.close()


        uu = torch.tensor(np.linspace(0, 3, 1000)).to(device)
        in_features = uu[:, None]
        with torch.no_grad():
            func = model.lin_edge(in_features.float())
            func = func[:, 0]
        uu = uu.to(dtype=torch.float32)
        func = func.to(dtype=torch.float32)

        text_trap = StringIO()
        sys.stdout = text_trap

        model_pysrr = PySRRegressor(
            niterations=30,  # < Increase me for better results
            binary_operators=["+", "*"],
            unary_operators=[
                "cos",
                "exp",
                "sin",
                "tanh"
            ],
            random_state=0,
            temp_equation_file=False
        )

        model_pysrr.fit(to_numpy(uu[:, None]), to_numpy(func[:, None]))

        sys.stdout = sys.__stdout__

        expr = model_pysrr.sympy(2).as_terms()[0]
        coeff = expr[0][1][0][0]
        print(expr)
        logger.info(expr)

        A = torch.zeros(n_particles, n_particles, device=device, requires_grad=False, dtype=torch.float32)
        if 'asymmetric' in config.simulation.adjacency_matrix:
            A = model.vals
        else:
            i, j = torch.triu_indices(n_particles, n_particles, requires_grad=False, device=device)
            A[i,j] = model.vals
            A.T[i,j] = model.vals

        fig, ax = fig_init()
        gt_weight = to_numpy(adjacency[adj_t])
        pred_weight = to_numpy(A[adj_t]) * coeff
        x_data = gt_weight
        y_data = pred_weight.squeeze()
        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
        residuals = y_data - linear_model(x_data, *lin_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
        plt.scatter(gt_weight, pred_weight, s=200, c='k', edgecolors='none')
        plt.ylabel('Learned $A_{ij}$ values', fontsize=64)
        plt.xlabel('True network $A_{ij}$ values', fontsize=64)
        print(f"R^2$: {np.round(r_squared, 3)}  Slope: {np.round(lin_fit[0], 2)}   offset: {np.round(lin_fit[1], 2)}  ")
        logger.info(f"R^2$: {np.round(r_squared, 3)}  Slope: {np.round(lin_fit[0], 2)}   offset: {np.round(lin_fit[1], 2)}  ")
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/Aij_{config_file}_{epoch}.tif", dpi=300)
        plt.close()

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(1,2,1)
        plt.imshow(to_numpy(adjacency), cmap='viridis', vmin=0, vmax=0.01)
        plt.title('True $A_{ij}$', fontsize=64)
        plt.xticks(fontsize=24.0)
        plt.yticks(fontsize=24.0)
        ax = fig.add_subplot(1,2,2)
        plt.imshow(to_numpy(model.vals)*coeff, cmap='viridis', vmin=0, vmax=0.01)
        plt.title('Learned $A_{ij}$', fontsize=64)
        plt.xticks(fontsize=24.0)
        plt.yticks(fontsize=24.0)
        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/Aij_comparison_{config_file}_{epoch}.tif", dpi=300)
        plt.close()

        fig, ax = fig_init()
        plt.scatter(to_numpy(adjacency),to_numpy(model.vals)*coeff)

        fig, ax = fig_init()
        gt_weight = to_numpy(adjacency)
        pred_weight = to_numpy(model.vals) * coeff
        x_data = np.reshape(gt_weight, (n_particles * n_particles))
        y_data =  np.reshape(pred_weight,  (n_particles * n_particles))
        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
        residuals = y_data - linear_model(x_data, *lin_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        plt.plot(x_data, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=4)
        plt.scatter(gt_weight, pred_weight, s=200, c='k', edgecolors='none')
        plt.ylabel('Learned $A_{ij}$ values', fontsize=64)
        plt.xlabel('True network $A_{ij}$ values', fontsize=64)
        print(f"R^2$: {np.round(r_squared, 3)}  Slope: {np.round(lin_fit[0], 2)}   offset: {np.round(lin_fit[1], 2)}  ")
        logger.info(f"R^2$: {np.round(r_squared, 3)}  Slope: {np.round(lin_fit[0], 2)}   offset: {np.round(lin_fit[1], 2)}  ")
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/all_Aij_{config_file}_{epoch}.tif", dpi=300)
        plt.close()

        true_func = torch.tanh(uu)
        fig, ax = fig_init()
        plt.xlabel(r'$u$', fontsize=78)
        plt.ylabel(r'$f(u)$', fontsize=78)
        plt.plot(to_numpy(uu), to_numpy(true_func), linewidth=20, c='g', label=r'true $f(u)$')
        plt.plot(to_numpy(uu), to_numpy(func) / coeff, linewidth=8, c='k', label=r'learned $f(u)$')
        plt.legend(fontsize=32.0, loc='upper right')
        plt.ylim([0,1.4])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/comparison_f_u_{config_file}_{epoch}.tif", dpi=300)
        plt.close()

        # Analysis of \Phi(u)

        for type in range(2):
            uu = torch.tensor(np.linspace(0, 4, 1000)).to(device)
            uu = uu.to(dtype=torch.float32)

            pos = np.argwhere(to_numpy(x[:, 5]) == type)
            learned_func = func_list[pos].squeeze()
            learned_func = torch.median(learned_func, dim=0).values
            p = config.simulation.params
            if len(p) > 1:
                p = torch.tensor(p, device=device)
            true_func = -to_numpy(uu) * to_numpy(p[type, 0]) + to_numpy(p[type, 1]) * np.tanh(to_numpy(uu))

            fig, ax = fig_init()
            if type==0:
                plt.plot(to_numpy(uu), true_func[:,None], linewidth=20, label='true', c='xkcd:sky blue')
            else:
                plt.plot(to_numpy(uu), true_func[:,None], linewidth=20, label='true', c='orange')
            plt.plot(to_numpy(uu), to_numpy(learned_func), linewidth=8, c='k', label='learned')
            plt.xlabel(r'$u$', fontsize=78)
            plt.ylabel(r'$\Phi_0(u)$', fontsize=78)
            plt.legend(fontsize=32.0)
            plt.ylim([-0.25, 0.25])
            plt.xlim([0, 3])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/results/comparison_phi_{type}_{config_file}_{epoch}.tif", dpi=300)
            plt.close()

            text_trap = StringIO()
            sys.stdout = text_trap

            model_pysrr = PySRRegressor(
                niterations=100,  # < Increase me for better results
                unary_operators=[
                    "tanh"
                ],
                nested_constraints={
                    "tanh": {"tanh": 0},
                },
                random_state=0,
                maxsize=20,
                maxdepth=6,
                temp_equation_file=False
            )
            model_pysrr.fit(to_numpy(uu[:, None]), to_numpy(learned_func[:, None]))

            sys.stdout = sys.__stdout__



        # model_pysrr = PySRRegressor(
        #     niterations=100,  # < Increase me for better results
        #     unary_operators=[
        #         "tanh"
        #     ],
        #     random_state=0,
        #     temp_equation_file=False,
        #     maxsize=20,
        #     maxdepth=6
        # )
        # model_pysrr.fit(to_numpy(uu[:, None]), true_func[:, None])









        # model_kan = KAN(width=[1, 1], grid=5, k=3, seed=0)
        # model_kan.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.)
        # lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']
        # model_kan.auto_symbolic(lib=lib)
        # model_kan.train(dataset, steps=20)
        # formula, variables = model_kan.symbolic_formula()
        # print(formula)
        #
        # model_kan = KAN(width=[1, 5, 1], grid=5, k=3, seed=0)
        # model_kan.train(dataset, opt="LBFGS", steps=50, lamb=0.01, lamb_entropy=10.)
        # model_kan = model_kan.prune()
        # model_kan.train(dataset, opt="LBFGS", steps=50);
        # for k in range(10):
        #     lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']
        #     model_kan.auto_symbolic(lib=lib)
        #     model_kan.train(dataset, steps=100)
        #     formula, variables = model_kan.symbolic_formula()
        #     print(formula)


def plot_agents(config_file, epoch_list, log_dir, logger, device):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Training data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    delta_t = simulation_config.delta_t
    noise_level = train_config.noise_level
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames

    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)
    n_runs = train_config.n_runs
    has_state = (config.simulation.state_type != 'discrete')

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))

    print('Create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    # net = f"./log/try_{config_file}/models/best_model_with_1_graphs_3.pt"
    # print(f'Loading existing model {net}...')
    # state_dict = torch.load(net,map_location=device)
    # model.load_state_dict(state_dict['model_state_dict'])

    print('Load data ...')
    time_series, signal = load_agent_data(dataset_name, device=device)

    velocities = [t.velocity for t in time_series]
    velocities.pop(0)  # the first element is always NaN
    velocities = torch.stack(velocities)
    if torch.any(torch.isnan(velocities)):
        raise ValueError('Discovered NaN in velocities. Aborting.')
    velocities = bc_dpos(velocities)

    positions = torch.stack([t.pos for t in time_series])
    min = torch.min(positions[:, :, 0])
    max = torch.max(positions[:, :, 0])
    mean = torch.mean(positions[:, :, 0])
    std = torch.std(positions[:, :, 0])
    print(f"min: {min}, max: {max}, mean: {mean}, std: {std}")

    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'))
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'))

    time.sleep(0.5)
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')

    n_particles = config.simulation.n_particles
    print(f'N particles: {n_particles}')
    logger.info(f'N particles:  {n_particles}')

    if os.path.exists(f'./log/try_{config_file}/edge_p_p_list.npz'):
        print('Load list of edges index ...')
        edge_p_p_list = np.load(f'./log/try_{config_file}/edge_p_p_list.npz')
    else:
        print('Create list of edges index ...')
        edge_p_p_list = []
        for k in trange(n_frames):
            time_point = time_series[k]
            x = bundle_fields(time_point, "pos", "velocity", "internal", "state", "reversal_timer").clone().detach()
            x = torch.column_stack((torch.arange(0, n_particles, device=device), x))

            nbrs = NearestNeighbors(n_neighbors=simulation_config.n_neighbors, algorithm='auto').fit(to_numpy(x[:, 1:dimension + 1]))
            distances, indices = nbrs.kneighbors(to_numpy(x[:, 1:dimension + 1]))
            edge_index = []
            for i in range(indices.shape[0]):
                for j in range(1, indices.shape[1]):  # Start from 1 to avoid self-loop
                    edge_index.append((i, indices[i, j]))
            edge_index = np.array(edge_index)
            edge_index = torch.tensor(edge_index, device=device).t().contiguous()
            edge_p_p_list.append(to_numpy(edge_index))
        np.savez(f'./log/try_{config_file}/edge_p_p_list', *edge_p_p_list)

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    for epoch in epoch_list:

        net = f"./log/try_{config_file}/models/best_model_with_0_graphs_{epoch}.pt"
        print(f'network: {net}')
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()


        for k in trange(2, n_frames-2,10):

            time_point = time_series[k]
            x = bundle_fields(time_point, "pos", "velocity", "internal", "state", "reversal_timer").clone().detach()
            x = torch.column_stack((torch.arange(0, n_particles, device=device), x))
            x[:, 1:5] = x[:, 1:5] / 1000

            edges = edge_p_p_list[f'arr_{k}']
            edges = torch.tensor(edges, dtype=torch.int64, device=device)
            dataset = data.Data(x=x[:, :], edge_index=edges)

            if model_config.prediction == 'first_derivative':
                time_point = time_series[k + 1]
                y = bc_dpos(time_point.velocity.clone().detach() / 1000)
            else:
                time_point = time_series[k + 1]
                v_prev = bc_dpos(time_point.velocity.clone().detach() / 1000)
                time_point = time_series[k - 1]
                v_next = bc_dpos(time_point.velocity.clone().detach() / 1000)
                y = (v_next - v_prev)

            embedding = to_numpy(model.a[0][k].squeeze())

            ax, fig = fig_init()
            plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), c = embedding,  s=0.1, alpha=1)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_recons/Fig_{k}.tif", dpi=87)
            plt.close()

            # in_features = torch.cat((torch.zeros((300000,7),device=device), embedding), dim=-1)
            # out = model.lin_edge(in_features.float())

        for x_ in [0,1]:
            for y_ in [0,1]:
                fig = plt.figure(figsize=(5, 5))
                plt.scatter(to_numpy(y[:, x_]), embedding[:, y_], s=0.1, c='k', alpha=0.01)
                plt.xlabel(f'$x_{x_}$', fontsize=48)
                plt.ylabel(f'$y_{y_}$', fontsize=48)
                plt.tight_layout()

        x = model.a[0][k].squeeze()


        model_kan = KAN(width=[2, 5, 2])
        dataset={}

        dataset['train_input'] = x[0:10000]
        dataset['test_input'] = x[12000:13000]
        dataset['train_label'] = y[0:10000]
        dataset['test_label'] = y[12000:13000]

        model_kan.train(dataset, opt="LBFGS", steps=200, lamb=0.01, lamb_entropy=10.)

        model_kan.plot()

        lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']
        model_kan.auto_symbolic(lib=lib)
        model_kan.train(dataset, steps=20)


        fig = plt.figure(figsize=(5, 5))
        plt.scatter(to_numpy(y[:, 1]), to_numpy(embedding[:, 0]), s=0.1, c='k', alpha=0.01)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.scatter(to_numpy(embedding[:, 0]), to_numpy(embedding[:, 1]), to_numpy(y[:, 0]), s=0.1, c='k', alpha=0.1)






        # if has_state:
        #     ax, fig = fig_init()
        #     embedding = torch.reshape(model.a[0], (n_particles * n_frames, 2))
        #     plt.scatter(to_numpy(embedding[:, 0]), to_numpy(embedding[:, 1]), s=0.1, alpha=0.01, c='k')
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.tight_layout()
        #     plt.savefig(f"./{log_dir}/tmp_training/embedding/Fig_{epoch}_{N}.tif", dpi=87)
        # else:
        #     ax, fig = fig_init()
        #     embedding = model.a[0]
        #     plt.scatter(to_numpy(embedding[:, 0]), to_numpy(embedding[:, 1]), s=1, alpha=0.1, c='k')
        #     plt.xticks([])
        #     plt.yticks([])
        #     plt.tight_layout()
        #     plt.savefig(f"./{log_dir}/tmp_training/embedding/Fig_{epoch}_{N}.tif", dpi=87)
        #

        # # plt.scatter(to_numpy(y[:, 1]), to_numpy(pred[:, 1]), s=0.1, alpha=0.1)
        # plt.xticks([])
        # plt.yticks([])
        # plt.tight_layout()
        # plt.savefig(f"./{log_dir}/tmp_training/particle/Fig_{epoch}_{N}.tif", dpi=87)


def data_video_validation(config_file, epoch_list, log_dir, logger, device):
    print('')

    # Load parameters from config file
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    print(f'Save movie ... {config.graph_model.particle_model_name} {config.graph_model.mesh_model_name}')

    graph_files = glob.glob(f"./graphs_data/graphs_{dataset_name}/generated_data/*")
    N_files = len(graph_files)
    recons_files = glob.glob(f"{log_dir}/tmp_recons/*")

    # import cv2
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter(f"video/validation_{dataset_name}.avi", fourcc, 20.0, (1024, 2048))

    os.makedirs(f"video_tmp/{config_file}", exist_ok=True)

    for n in trange(N_files):
        generated = imread(graph_files[n])
        reconstructed = imread(recons_files[n])
        frame = np.concatenate((generated[:, :, 0:3], reconstructed[:, :, 0:3]), axis=1)
        # out.write(frame)
        imsave(f"video_tmp/{config_file}/{dataset_name}_{10000 + n}.tif", frame)

    # Release the video writer
    # out.release()

    # print("Video saved as 'output.avi'")


def data_video_training(config_file, epoch_list, log_dir, logger, device):
    print('')

    # Load parameters from config file
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    dataset_name = config.dataset

    max_radius = config.simulation.max_radius
    if config.graph_model.particle_model_name != '':
        config_model = config.graph_model.particle_model_name
    elif config.graph_model.signal_model_name != '':
        config_model = config.graph_model.signal_model_name
    elif config.graph_model.mesh_model_name != '':
        config_model = config.graph_model.mesh_model_name

    print(f'Save movie ... {config.graph_model.particle_model_name} {config.graph_model.mesh_model_name}')

    embedding = imread(f"{log_dir}/embedding.tif")
    function = imread(f"{log_dir}/function.tif")
    # field = imread(f"{log_dir}/field.tif")

    matplotlib.use("Qt5Agg")

    os.makedirs(f"video_tmp/{config_file}_training", exist_ok=True)

    for n in trange(embedding.shape[0]):
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(embedding[n, :, :, 0:3])
        plt.xlabel(r'$a_{i0}$', fontsize=32)
        plt.ylabel(r'$a_{i1}$', fontsize=32)
        plt.xticks([])
        plt.yticks([])
        match config_file:
            case 'wave_slit':
                if n < 50:
                    plt.text(0, 1.1, f'epoch = 0,   it = {n * 200}', ha='left', va='top', transform=ax.transAxes,
                             fontsize=32)
                else:
                    plt.text(0, 1.1, f'Epoch={n - 49}', ha='left', va='top', transform=ax.transAxes, fontsize=32)
            case 'arbitrary_3':
                if n < 17:
                    plt.text(0, 1.1, f'epoch = 0,   it = {n * 200}', ha='left', va='top', transform=ax.transAxes,
                             fontsize=32)
                else:
                    plt.text(0, 1.1, f'epoch = {n - 16}', ha='left', va='top', transform=ax.transAxes, fontsize=32)
            case 'arbitrary_3_field_video_bison_siren_with_time':
                if n < 13 * 3:
                    plt.text(0, 1.1, f'epoch= {n // 13} ,   it = {(n % 13) * 500}', ha='left', va='top',
                             transform=ax.transAxes, fontsize=32)
                else:
                    plt.text(0, 1.1, f'epoch = {n - 13 * 3 + 3}', ha='left', va='top', transform=ax.transAxes,
                             fontsize=32)
            case 'arbitrary_64_256':
                if n < 51:
                    plt.text(0, 1.1, f'epoch= 0 ,   it = {n * 200}', ha='left', va='top', transform=ax.transAxes,
                             fontsize=32)
                else:
                    plt.text(0, 1.1, f'epoch = {n - 50}', ha='left', va='top', transform=ax.transAxes, fontsize=32)
            case 'boids_16_256' | 'gravity_16':
                if n < 50:
                    plt.text(0, 1.1, f'epoch = 0,   it = {n * 200}', ha='left', va='top', transform=ax.transAxes,
                             fontsize=32)
                else:
                    plt.text(0, 1.1, f'Epoch={n - 49}', ha='left', va='top', transform=ax.transAxes, fontsize=32)

        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(function[n, :, :, 0:3])
        # plt.ylabel(r'$f(a_i,d_{ij})$', fontsize=32)
        # plt.xlabel(r'$d_{ij}$', fontsize=32)
        plt.ylabel('x', fontsize=32)
        plt.xlabel('y', fontsize=32)
        plt.xticks(fontsize=16.0)
        plt.yticks(fontsize=16.0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        # fmt = lambda x, pos: '{:.3f}'.format(x / 1000 * max_radius, pos)
        # ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))

        match config_file:
            case 'wave_slit':
                fmt = lambda x, pos: '{:.1f}'.format((x / 1000), pos)
                ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                fmt = lambda x, pos: '{:.1f}'.format((1 - x / 1000), pos)
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            case 'arbitrary_3_field_video_bison_siren_with_time':
                fmt = lambda x, pos: '{:.2f}'.format(-x / 1000 * 0.7 + 0.3, pos)
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            case 'arbitrary_3':
                fmt = lambda x, pos: '{:.2f}'.format(-x / 1000 * 0.7 + 0.3, pos)
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            case 'arbitrary_64_256':
                fmt = lambda x, pos: '{:.2f}'.format(-x / 1000 * 0.7 + 0.3, pos)
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            case 'boids_16_256':
                fmt = lambda x, pos: '{:.2f}e-4'.format((-x / 1000 + 0.5) * 2, pos)
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            case 'boids_16_256' | 'gravity_16':
                fmt = lambda x, pos: '{:.1f}e5'.format((1 - x / 1000) * 5, pos)
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))

        # ax = fig.add_subplot(1, 3, 3)
        # ax.imshow(field[n, :, :, 0:3],cmap='grey')

        plt.tight_layout()
        plt.tight_layout()
        plt.savefig(f"video_tmp/{config_file}_training/training_{config_file}_{10000 + n}.tif", dpi=64)
        plt.close()

    # plt.text(0, 1.05, f'Frame {it}', ha='left', va='top', transform=ax.transAxes, fontsize=32)
    # ax.tick_params(axis='both', which='major', pad=15)


def data_plot(config, config_file, epoch_list, device):
    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    matplotlib.rcParams['savefig.pad_inches'] = 0

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))
    print('log_dir: {}'.format(log_dir))

    logging.basicConfig(filename=f'{log_dir}/results.log', format='%(asctime)s %(message)s', filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    os.makedirs(os.path.join(log_dir, 'results'), exist_ok=True)

    if config.training.sparsity != 'none':
        print(
            f'GNN trained with simulation {config.graph_model.particle_model_name} ({config.simulation.n_particle_types} types), with cluster method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
        logger.info(
            f'GNN trained with simulation {config.graph_model.particle_model_name} ({config.simulation.n_particle_types} types), with cluster method: {config.training.cluster_method}   threshold: {config.training.cluster_distance_threshold}')
    else:
        print(
            f'GNN trained with simulation {config.graph_model.particle_model_name} ({config.simulation.n_particle_types} types), no clustering')
        logger.info(
            f'GNN trained with simulation {config.graph_model.particle_model_name} ({config.simulation.n_particle_types} types), no clustering')

    if os.path.exists(f'{log_dir}/loss.pt'):
        loss = torch.load(f'{log_dir}/loss.pt')
        fig, ax = fig_init(formatx='%.0f', formaty='%.5f')
        plt.plot(loss, color='k', linewidth=4)
        plt.xlim([0, 20])
        plt.ylabel('Loss', fontsize=78)
        plt.xlabel('Epochs', fontsize=78)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/results/loss_{config_file}.tif", dpi=170.7)
        plt.close()
        print('final loss {:.3e}'.format(loss[-1]))
        logger.info('final loss {:.3e}'.format(loss[-1]))

    match config.graph_model.particle_model_name:
        case 'PDE_Agents_A' | 'PDE_Agents_B':
            plot_agents(config_file, epoch_list, log_dir, logger, device)
        case 'PDE_Cell_A' | 'PDE_Cell_B':
            if config.simulation.has_cell_state:
                plot_cell_state(config_file, epoch_list, log_dir, logger, device)
        case 'PDE_A':
            if config.simulation.non_discrete_level>0:
                plot_attraction_repulsion_continuous(config_file, epoch_list, log_dir, logger, device)
            elif config.training.do_tracking:
                plot_attraction_repulsion_tracking(config_file, epoch_list, log_dir, logger, device)
            else:
                plot_attraction_repulsion(config_file, epoch_list, log_dir, logger, device)
        case 'PDE_A_bis':
            plot_attraction_repulsion_asym(config_file, epoch_list, log_dir, logger, device)
        case 'PDE_B' | 'PDE_Cell_B':
            plot_boids(config_file, epoch_list, log_dir, logger, device)
        case 'PDE_ParticleField_B' | 'PDE_ParticleField_A':
            plot_particle_field(config_file, epoch_list, log_dir, logger, 'grey', device)
        case 'PDE_E':
            plot_Coulomb(config_file, epoch_list, log_dir, logger, device)
        case 'PDE_G':
            if config_file == 'gravity_continuous':
                plot_gravity_continuous(config_file, epoch_list, log_dir, logger, device)
            else:
                plot_gravity(config_file, epoch_list, log_dir, logger, device)

    match config.graph_model.mesh_model_name:
        case 'WaveMesh':
            plot_wave(config_file=config_file, epoch_list=epoch_list, log_dir=log_dir, logger=logger, cc='viridis',
                           device=device)
        case 'RD_RPS_Mesh':
            plot_RD_RPS(config_file=config_file, epoch_list=epoch_list, log_dir=log_dir, logger=logger, cc='viridis',
                           device=device)

    if 'PDE_N' in config.graph_model.signal_model_name:
        plot_signal(config_file, epoch_list, log_dir, logger, 'viridis', device)

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def get_figures(index):

    epoch_list = ['20']
    match index:
        case '3':
            config_list = ['arbitrary_3_continuous', 'arbitrary_3', 'arbitrary_3_3', 'arbitrary_16', 'arbitrary_32','arbitrary_64']
        case '4':
            config_list = ['arbitrary_3_field_video_bison_quad']
        case 'supp1':
            config_list = ['arbitrary_3']
            epoch_list= ['0_0', '0_200', '0_1000', '20']
        case 'supp4':
            config_list = ['arbitrary_16', 'arbitrary_16_noise_0_3', 'arbitrary_16_noise_0_4', 'arbitrary_16_noise_0_5']
        case 'supp5':
            config_list = ['arbitrary_3_dropout_30', 'arbitrary_3_dropout_10', 'arbitrary_3_dropout_10_no_ghost']
        case 'supp6':
            config_list = ['arbitrary_3_field_boats']
        case 'supp7':
            config_list = ['gravity_16']
            epoch_list= ['0_0', '0_5000', '1_0', '20']
        case 'supp8':
            config_list = ['gravity_16', 'gravity_continuous', 'Coulomb_3_256']
        case 'supp9':
            config_list = ['gravity_16_noise_0_4', 'Coulomb_3_noise_0_4', 'Coulomb_3_noise_0_3', 'gravity_16_noise_0_3']
        case 'supp10':
            config_list = ['gravity_16_dropout_10', 'gravity_16_dropout_30', 'Coulomb_3_dropout_10_no_ghost', 'Coulomb_3_dropout_10']
        case 'supp11':
            config_list = ['boids_16_256']
            epoch_list = ['0_0', '0_2000', '0_10000', '20']
        case 'supp12':
            config_list = ['boids_16_256', 'boids_32_256', 'boids_64_256']
        case 'supp14':
            config_list = ['boids_16_noise_0_3', 'boids_16_noise_0_4', 'boids_16_dropout_10', 'boids_16_dropout_10_no_ghost']
        case 'supp15':
            config_list = ['wave_slit_ter']
            epoch_list = ['20', '0_1600', '1', '5']
        case 'supp16':
            config_list = ['wave_boat_ter']
            epoch_list = ['20', '0_1600', '1', '5']
        case 'supp17':
            config_list = ['RD_RPS']
        case 'supp18':
            config_list = ['signal_N_100_2_a']
        case _:
            config_list = ['arbitrary_3']


    match index:
        case '3' | '4' | 'supp4' | 'supp5' | 'supp6' | 'supp7' | 'supp8' | 'supp9' | 'supp10' | 'supp11' | 'supp12' | 'supp15' |'supp16' |'supp18':
            for config_file in config_list:
                config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
                data_plot(config=config, config_file=config_file, epoch_list=epoch_list, device=device)
                data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                                  best_model=20, run=0, step=64, test_simulation=False,
                                  sample_embedding=False, device=device)  # config.simulation.n_frames // 7
                print(' ')
                print(' ')

        case 'supp1':
            config = ParticleGraphConfig.from_yaml(f'./config/arbitrary_3.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True, bSave=True, step=config.simulation.n_frames // 3)
            config = ParticleGraphConfig.from_yaml(f'./config/arbitrary_3_bis.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=0, style='latex bw', alpha=1, erase=True, bSave=True, step=config.simulation.n_frames // 3)
            for config_file in config_list:
                config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
                data_plot(config=config, config_file=config_file, epoch_list=epoch_list, device=device)
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                              best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                              sample_embedding=False, device=device)

        case 'supp2':
            config_file = 'arbitrary_3_bis'
            config = ParticleGraphConfig.from_yaml(f'./config/arbitrary_3_bis.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True,
                          scenario='stripes', ratio = 1, bSave=True, step=config.simulation.n_frames // 3)
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                      sample_embedding=False, device=device)
            config_file = 'arbitrary_3_ter'
            config = ParticleGraphConfig.from_yaml(f'./config/arbitrary_3_ter.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True,
                          scenario='pattern', ratio = 1, bSave=True, step=config.simulation.n_frames // 3)
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                      sample_embedding=True, device=device)
            config_file = 'arbitrary_3_quad'
            config = ParticleGraphConfig.from_yaml(f'./config/arbitrary_3_quad.yaml')
            # data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True,
            #               scenario='pattern', ratio = 3, bSave=True, step=config.simulation.n_frames // 3)
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                      sample_embedding=True, ratio = 3, device=device)

        case 'supp3':
            config_file = 'arbitrary_3_bis'
            config = ParticleGraphConfig.from_yaml(f'./config/arbitrary_3_bis.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True,
                          scenario='uniform 0', ratio = 3, bSave=True, step=config.simulation.n_frames // 3)
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                      sample_embedding=True, device=device)
            config_file = 'arbitrary_3_ter'
            config = ParticleGraphConfig.from_yaml(f'./config/arbitrary_3_ter.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True,
                          scenario='uniform 1', ratio = 3, bSave=True, step=config.simulation.n_frames // 3)
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                      sample_embedding=True, device=device)
            config_file = 'arbitrary_3_quad'
            config = ParticleGraphConfig.from_yaml(f'./config/arbitrary_3_quad.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True,
                          scenario='uniform 2', ratio = 3, bSave=True, step=config.simulation.n_frames // 3)
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                      sample_embedding=True, ratio = 3, device=device)

        case 'supp6':
            config_file = 'arbitrary_3_field_boats'
            config = ParticleGraphConfig.from_yaml(f'./config/arbitrary_3_field_boats.yaml')
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                      sample_embedding=False, device=device)

        case 'supp7':
            config = ParticleGraphConfig.from_yaml(f'./config/gravity_16_bis.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex bw', alpha=1, erase=True,
                          scenario='stripes', ratio=1, bSave=True, step=config.simulation.n_frames // 3)
            config_file = 'gravity_16'
            config = ParticleGraphConfig.from_yaml(f'./config/gravity_16.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True,
                          scenario='stripes', ratio=1, bSave=True, step=config.simulation.n_frames // 3)
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                      sample_embedding=False, device=device)

        case 'supp11':
            config = ParticleGraphConfig.from_yaml(f'./config/boids_16_256_bis.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex bw', alpha=1, erase=True,
                          scenario='', ratio=1, bSave=True, step=config.simulation.n_frames // 3)
            config_file = 'boids_16_256'
            config = ParticleGraphConfig.from_yaml(f'./config/boids_16_256.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True,
                          scenario='', ratio=1, bSave=True, step=config.simulation.n_frames // 3)
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                      sample_embedding=False, device=device)

        case 'supp13':

            r=[]
            for n in range(16):
                result = np.load(f'./log/try_boids_16_256_{n}/rmserr_geomloss_boids_16_256_{n}.npy')
                print (n,result)
                r.append(result)
            print('mean',np.mean(r,axis=0))

            config = ParticleGraphConfig.from_yaml(f'./config/boids_16_256_bis.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1,
                          erase=True,
                          scenario='stripes', ratio=4, bSave=True, step=config.simulation.n_frames // 7)
            config_file = f'boids_16_256_bis'
            config = ParticleGraphConfig.from_yaml(f'./config/boids_16_256_bis.yaml')
            data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 7, test_simulation=False,
                      sample_embedding=True, device=device)

            for n in range(16):
                copyfile(f'./config/boids_16_256.yaml', f'./config/boids_16_256_{n}.yaml')
                config_file = f'boids_16_256_{n}'
                config = ParticleGraphConfig.from_yaml(f'./config/boids_16_256_{n}.yaml')
                data_generate(config, device=device, visualize=True, run_vizualized=1, style='no_ticks color', alpha=1, erase=True,
                              scenario=f'uniform {n}', ratio=4, bSave=True, step=config.simulation.n_frames // 3)
                data_test(config=config, config_file=config_file, visualize=True, style='no_ticks color', verbose=False,
                          best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                          sample_embedding=True, device=device)

        case 'supp15':
            config = ParticleGraphConfig.from_yaml(f'./config/wave_slit_ter.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True,
                          scenario='', ratio=1, bSave=True, step=config.simulation.n_frames // 3)
            config_file = 'wave_slit_bis'
            config = ParticleGraphConfig.from_yaml(f'./config/wave_slit_bis.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1,
                          erase=True,
                          scenario='', ratio=1, bSave=True, step=config.simulation.n_frames // 100)
            config_file = 'wave_slit_bis'
            config = ParticleGraphConfig.from_yaml(f'./config/wave_slit_bis.yaml')
            data_test(config=config, config_file=config_file, visualize=True, style='latex color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 100, test_simulation=False,
                      sample_embedding=False, device=device)

        case 'supp16':
            config = ParticleGraphConfig.from_yaml(f'./config/wave_boat_ter.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True,
                          scenario='', ratio=1, bSave=True, step=config.simulation.n_frames // 3)
            config_file = 'wave_boat_bis'
            config = ParticleGraphConfig.from_yaml(f'./config/wave_boat_bis.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1,
                          erase=True,
                          scenario='', ratio=1, bSave=True, step=config.simulation.n_frames // 3)
            config_file = 'wave_boat_bis'
            config = ParticleGraphConfig.from_yaml(f'./config/wave_boat_bis.yaml')
            data_test(config=config, config_file=config_file, visualize=True, style='latex color', verbose=False,
                      best_model=20, run=1, step=config.simulation.n_frames // 100, test_simulation=False,
                      sample_embedding=False, device=device)

        case 'supp17':
            config = ParticleGraphConfig.from_yaml(f'./config/RD_RPS.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True, bSave=True, step=config.simulation.n_frames // 3)
            config = ParticleGraphConfig.from_yaml(f'./config/RD_RPS_bis.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True, bSave=True, step=config.simulation.n_frames // 3)
            for config_file in config_list:
                config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
                data_plot(config=config, config_file=config_file, epoch_list=epoch_list, device=device)
            data_test(config=config, config_file=config_file, visualize=True, style='latex color', verbose=False,
                              best_model=20, run=1, step=config.simulation.n_frames // 3, test_simulation=False,
                              sample_embedding=False, device=device)

        case 'supp18':
            config = ParticleGraphConfig.from_yaml(f'./config/signal_N_100_2.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True, bSave=True, step=config.simulation.n_frames // 3)
            config = ParticleGraphConfig.from_yaml(f'./config/signal_N_100_2_a.yaml')
            data_generate(config, device=device, visualize=True, run_vizualized=1, style='latex color', alpha=1, erase=True, bSave=True, step=config.simulation.n_frames // 3)
            for config_file in config_list:
                config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
                data_plot(config=config, config_file=config_file, epoch_list=epoch_list, device=device)
            data_test(config=config, config_file=config_file, visualize=True, style=' color', verbose=False,
                              best_model=20, run=0, step=config.simulation.n_frames // 100, test_simulation=False,
                              sample_embedding=False, device=device)

    print(' ')
    print(' ')

    return config_list,epoch_list


if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(' ')
    print(f'device {device}')
    print(' ')

    try:
        matplotlib.use("Qt5Agg")
    except:
        pass

    f_list = ['supp15']
    for f in f_list:
        config_list,epoch_list = get_figures(f)




    # config_list = ["arbitrary_3_cell_sequence_d_bis"]
    # config_list = ["arbitrary_3_cell_sequence_f"]
    # # config_list = ['signal_N_100_2_d']
    # config_list = ['signal_N_100_2_a']
    # config_list = ['boids_division_model_f2']
    # config_list = ["agents_e"]
    # config_list = ["arbitrary_division_model_passive_v"]

    # for config_file in config_list:
    #     config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    #     data_plot(config=config, config_file=config_file, epoch_list=['40_0'], device=device)
    #     # plot_generated(config=config, run=0, style='color', step = 2, device=device)
    #     # plot_focused_on_cell(config=config, run=0, style='color', cell_id=175, step = 5, device=device)





