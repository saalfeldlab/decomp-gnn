import numpy as np
import torch
import torch.nn as nn
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy
from ParticleGraph.models import Siren_Network


class Interaction_Particle_Field(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Model learning the acceleration of particles as a function of their relative distance and relative velocities.
    The interaction function is defined by a MLP self.lin_edge
    The particle embedding is defined by a table self.a

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the acceleration of the particles (dimension 2)
    """

    def __init__(self, config, device, aggr_type=None, bc_dpos=None, dimension=2):

        super(Interaction_Particle_Field, self).__init__(aggr=aggr_type)  # "Add" aggregation.

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device
        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_mp_layers
        self.n_particles = simulation_config.n_particles
        self.n_nodes = simulation_config.n_nodes
        self.n_nodes_per_axis = int(np.sqrt(self.n_nodes))
        self.max_radius = simulation_config.max_radius
        self.data_augmentation = train_config.data_augmentation
        self.noise_level = train_config.noise_level
        self.embedding_dim = model_config.embedding_dim
        self.n_dataset = train_config.n_runs
        self.prediction = model_config.prediction
        self.update_type = model_config.update_type
        self.n_layers_update = model_config.n_layers_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.sigma = simulation_config.sigma
        self.model = model_config.particle_model_name
        self.bc_dpos = bc_dpos
        self.n_ghosts = int(train_config.n_ghosts)
        self.dimension = dimension

        if train_config.large_range:
            self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device, activation='tanh')
        else:
            self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)

        if simulation_config.has_cell_division :
            self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, 20500, 2)), device=self.device,
                             requires_grad=True, dtype=torch.float32))
        else:
            self.a = nn.Parameter(
                torch.tensor(np.ones((self.n_dataset, int(self.n_particles) + self.n_ghosts, self.embedding_dim)), device=self.device,
                             requires_grad=True, dtype=torch.float32))

        if (model_config.field_method == 'tensor'):
            self.field = nn.Parameter(
                    torch.tensor(np.zeros((self.n_dataset, int(self.n_nodes), 1)), device=self.device, requires_grad=True, dtype=torch.float32))
        elif (model_config.field_method == 'Siren_wo_time'):
            self.field=[]
            for n in range(self.n_dataset):
                image_width = self.n_nodes_per_axis
                self.field.append(Siren_Network(image_width=image_width, in_features=2, out_features=1, hidden_features=256, hidden_layers=8, outermost_linear=True, device=device, first_omega_0=80, hidden_omega_0=80.))
        elif (model_config.field_method == 'Siren_with_time'):
            self.field = []
            for n in range(self.n_dataset):
                image_width = self.n_nodes_per_axis
                self.field.append(Siren_Network(image_width=image_width, in_features=2, out_features=1, hidden_features=256, hidden_layers=8, outermost_linear=True, device=device, first_omega_0=80, hidden_omega_0=80.))



        if self.update_type != 'none':
            self.lin_update = MLP(input_size=self.output_size + self.embedding_dim + 2, output_size=self.output_size,
                                  nlayers=self.n_layers_update, hidden_size=self.hidden_dim_update, device=self.device)

    def forward(self, data=[], data_id=[], training=[], vnorm=[], phi=[], has_field=False):


        self.data_id = data_id
        self.vnorm = vnorm
        self.cos_phi = torch.cos(phi)
        self.sin_phi = torch.sin(phi)
        self.has_field = has_field
        self.training = training

        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        pos = x[:, 1:self.dimension+1]
        d_pos = x[:, self.dimension+1:1+2*self.dimension]
        if has_field:
            field = x[:,6:7]
        else:
            field = torch.ones_like(x[:,6:7])

        particle_id = x[:, 0:1]

        pred = self.propagate(edge_index, pos=pos, d_pos=d_pos, particle_id=particle_id, field=field)

        if self.update_type == 'linear':
            embedding = self.a[self.data_id, particle_id, :]
            pred = self.lin_update(torch.cat((pred, x[:, 3:5], embedding), dim=-1))

        return pred

    def message(self, pos_i, pos_j, d_pos_i, d_pos_j, particle_id_i, particle_id_j, field_j):
        # squared distance
        r = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1)) / self.max_radius
        delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius
        dpos_x_i = d_pos_i[:, 0] / self.vnorm
        dpos_y_i = d_pos_i[:, 1] / self.vnorm
        dpos_x_j = d_pos_j[:, 0] / self.vnorm
        dpos_y_j = d_pos_j[:, 1] / self.vnorm
        if self.dimension == 3:
            dpos_z_i = d_pos_i[:, 2] / self.vnorm
            dpos_z_j = d_pos_j[:, 2] / self.vnorm

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
        # embedding_j = self.a[self.data_id, to_numpy(particle_id_j), :].squeeze()

        match self.model:

            case 'PDE_ParticleField_A':
                    in_features = torch.cat((delta_pos, r[:, None], embedding_i), dim=-1)
            case 'PDE_ParticleField_B':
                    in_features = torch.cat(
                        (delta_pos, r[:, None], dpos_x_i[:, None], dpos_y_i[:, None], dpos_x_j[:, None],
                         dpos_y_j[:, None], embedding_i), dim=-1)

        out = self.lin_edge(in_features) * field_j


        return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p1, p2):

        if (self.model == 'PDE_A') | (self.model =='PDE_A_bis')  | (self.model=='PDE_ParticleField_A'):
            return r * (p1[0] * torch.exp(-r ** (2 * p1[1]) / (2 * self.sigma ** 2)) - p1[2] * torch.exp(-r ** (2 * p1[3]) / (2 * self.sigma ** 2)))
        if self.model == 'PDE_B':
            cohesion = p1[0] * 0.5E-5 * r
            separation = -p1[2] * 1E-8 / r
            return (cohesion + separation) * p1[1] / 500
        if self.model == 'PDE_G':
            psi = p1 / r ** 2
            return psi[:, None]
        if self.model == 'PDE_E':
            acc = p1 * p2 / r ** 2
            return -acc  # Elec particles
