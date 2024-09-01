import numpy as np
import torch
import torch.nn as nn
from ParticleGraph.models.MLP import MLP
from ParticleGraph.utils import to_numpy
import torch_geometric as pyg


class Mesh_Laplacian(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Model learning the second derivative of a scalar field on a mesh.
    The node embedding is defined by a table self.a
    Note the Laplacian coeeficients are in data.edge_attr

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the second derivative of a scalar field on a mesh (dimension 1).
    """

    def __init__(self, aggr_type=None, config=None, device=None, bc_dpos=None):
        super(Mesh_Laplacian, self).__init__(aggr=aggr_type)

        simulation_config = config.simulation
        model_config = config.graph_model

        self.device = device
        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_mp_layers
        self.embedding = model_config.embedding_dim
        self.n_particles = simulation_config.n_particles
        self.n_datasets = config.training.n_runs
        self.bc_dpos = bc_dpos

        self.lin_phi = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                           hidden_size=self.hidden_dim, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((int(self.n_datasets), int(self.n_particles), self.embedding)), device=self.device,
                         requires_grad=True, dtype=torch.float32))

    def forward(self, data, data_id):
        self.data_id = data_id
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        # deg = pyg_utils.degree(edge_index[0], data.num_nodes)

        u = x[:, 6:7]

        laplacian_u = self.propagate(edge_index, u=u, discrete_laplacian=edge_attr)

        particle_id = to_numpy(x[:, 0])
        embedding = self.a[self.data_id, particle_id, :]
        pred = self.lin_phi(torch.cat((laplacian_u, embedding), dim=-1))

        self.laplacian_u = laplacian_u

        # pos = to_numpy(data.x)
        # deg = pyg_utils.degree(edge_index[0], data.num_nodes)
        # plt.ion()
        # plt.scatter(pos[:,1],pos[:,2], s=20, c=to_numpy(deg),vmin=7,vmax=10)

        return pred

    def message(self, u_j, discrete_laplacian):
        L = discrete_laplacian[:,None] * u_j

        return L

    def update(self, aggr_out):
        return aggr_out  # self.lin_node(aggr_out)

    def psi(self, r, p):
        return p * r
