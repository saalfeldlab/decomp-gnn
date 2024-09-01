import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy


class PDE_A(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the speed of particles as a function of their relative position according to an attraction-repulsion law.
    The latter is defined by four parameters p = (p1, p2, p3, p4) and a parameter sigma.

    See https://github.com/gpeyre/numerical-tours/blob/master/python/ml_10_particle_system.ipynb

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the speed of the particles (dimension 2)
    """

    def __init__(self, aggr_type=[], p=[], sigma=[], bc_dpos=[], dimension=2):
        super(PDE_A, self).__init__(aggr=aggr_type)  # "mean" aggregation.

        self.p = p
        self.sigma = sigma
        self.bc_dpos = bc_dpos
        self.dimension = dimension

    def forward(self, data=[], has_field=False):
        x, edge_index = data.x, data.edge_index

        if has_field:
            field = x[:,6:7]
        else:
            field = torch.ones_like(x[:,0:1])

        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        particle_type = to_numpy(x[:, 1 + 2*self.dimension])
        parameters = self.p[particle_type,:]
        d_pos = self.propagate(edge_index, pos=x[:, 1:self.dimension+1], parameters=parameters, field=field)
        return d_pos


    def message(self, pos_i, pos_j, parameters_i, field_j):

        distance_squared = torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1)  # squared distance
        f = (parameters_i[:, 0] * torch.exp(-distance_squared ** parameters_i[:, 1] / (2 * self.sigma ** 2))
               - parameters_i[:, 2] * torch.exp(-distance_squared ** parameters_i[:, 3] / (2 * self.sigma ** 2)))
        d_pos = f[:, None] * self.bc_dpos(pos_j - pos_i) * field_j

        return d_pos

    def psi(self, r, p):
        return r * (p[0] * torch.exp(-r ** (2 * p[1]) / (2 * self.sigma ** 2))
                    - p[2] * torch.exp(-r ** (2 * p[3]) / (2 * self.sigma ** 2)))
