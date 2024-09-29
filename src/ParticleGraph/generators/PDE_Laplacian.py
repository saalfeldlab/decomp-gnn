import torch_geometric as pyg
from ParticleGraph.utils import to_numpy
import torch_geometric.utils as pyg_utils
import matplotlib.pyplot as plt


class PDE_Laplacian(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the Laplacian of a scalar field.

    Inputs
    ----------
    data : a torch_geometric.data object
    note the Laplacian coeeficients are in data.edge_attr

    Returns
    -------
    laplacian : float
        the Laplacian
    """

    def __init__(self, aggr_type=[], beta=[], bc_dpos=[]):
        super(PDE_Laplacian, self).__init__(aggr='add')  # "mean" aggregation.

        self.beta = beta
        self.bc_dpos = bc_dpos
        self.coeff = []

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # if self.coeff == []:
        #     particle_type = to_numpy(x[:, 5])
        #     c = self.c[particle_type]
        #     c = c[:, None]
        # else:

        c = self.coeff
        u = x[:, 6:7]

        laplacian_u = self.propagate(edge_index, u=u, edge_attr=edge_attr)
        dd_u = self.beta * c * laplacian_u

        self.laplacian_u = laplacian_u

        return dd_u

        pos = to_numpy(data.x)
        deg = pyg_utils.degree(edge_index[0], data.num_nodes)
        plt.ion()
        plt.scatter(pos[:,1],pos[:,2], s=20, c=to_numpy(deg),vmin=7,vmax=10)

    def message(self, u_j, edge_attr):
        L = edge_attr[:,None] * u_j

        return L

    def psi(self, I, p):
        return I
