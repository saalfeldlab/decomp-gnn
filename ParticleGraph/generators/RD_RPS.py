import torch
import torch_geometric as pyg
from ParticleGraph.utils import to_numpy


class RD_RPS(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Compute the reaction diffusion according to the rock paper scissor model.

    Inputs
    ----------
    data : a torch_geometric.data object
    Note the Laplacian coeeficients are in data.edge_attr

    Returns
    -------
    increment : float
        the first derivative of three scalar fields u, v and w
        
    """

    def __init__(self, aggr_type=[], bc_dpos=[]):
        super(RD_RPS, self).__init__(aggr='add')  # "mean" aggregation.

        self.bc_dpos = bc_dpos
        self.coeff = []

        self.a = 0.6

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # if self.coeff == []:
        #     particle_type = to_numpy(x[:, 5])
        #     c = self.c[particle_type]
        #     c = c[:, None]
        # else:

        c = self.coeff

        uvw = data.x[:, 6:9]
        laplace_uvw = c * self.propagate(data.edge_index, uvw=uvw, discrete_laplacian=data.edge_attr)
        p = torch.sum(uvw, axis=1)

        # This is equivalent to the nonlinear reaction diffusion equation:
        #   du = D * laplace_u + u * (1 - p - a * v)
        #   dv = D * laplace_v + v * (1 - p - a * w)
        #   dw = D * laplace_w + w * (1 - p - a * u)
        d_uvw = laplace_uvw + uvw * (1 - p[:,None] - self.a * uvw[:, [1, 2, 0]])

        return d_uvw

    def message(self, uvw_j, discrete_laplacian):
        return discrete_laplacian[:,None] * uvw_j

    def psi(self, I, p):
        return I
