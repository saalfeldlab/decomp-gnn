import torch
import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from ParticleGraph.utils import to_numpy


class PDE_E(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the acceleration of charged particles as a function of their relative position according to the Coulomb law.

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the acceleration of the particles (dimension 2)
    """

    def __init__(self, aggr_type=[], p=[], clamp=[], pred_limit=[], prediction=[], bc_dpos=[]):
        super(PDE_E, self).__init__(aggr='add')  # "mean" aggregation.

        self.p = p
        self.clamp = clamp
        self.pred_limit = pred_limit
        self.prediction = prediction
        self.bc_dpos = bc_dpos

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_index, _ = pyg_utils.remove_self_loops(edge_index)
        particle_type = to_numpy(x[:, 5])
        charge = self.p[particle_type]
        dd_pos = self.propagate(edge_index, pos=x[:,1:3], charge=charge[:, None])
        return dd_pos

    def message(self, pos_i, pos_j, charge_i, charge_j):
        distance_ij = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, axis=1))
        direction_ij = self.bc_dpos(pos_j - pos_i) / distance_ij[:,None]
        dd_pos = - charge_i * charge_j * direction_ij / (distance_ij[:,None] ** 2)
        return dd_pos

    def psi(self, r, p1, p2):
        dd_pos = p1 * p2 / r ** 2
        return -dd_pos  # Elec particles
