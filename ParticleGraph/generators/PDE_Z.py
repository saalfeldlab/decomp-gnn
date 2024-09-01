import torch
from torch_geometric.nn import MessagePassing


class PDE_Z(MessagePassing):
    """
    Dummy class that returns 0 as the acceleration of the particles.

    Returns
    -------
    pred : float
        an array of zeros (dimension 2)
    """

    def __init__(self, device=[]):
        super(PDE_Z, self).__init__(aggr='add')
        self.device = device
        self.p = torch.tensor([0, 0], device=device)

    def forward(self, data):

        return torch.zeros((data.x.shape[0],2), device=self.device, dtype=torch.float32)
