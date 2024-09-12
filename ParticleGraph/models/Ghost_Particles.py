import numpy as np
import torch
import torch.nn as nn

class Ghost_Particles(torch.nn.Module):

    def __init__(self, model_config, n_particles, vnorm, device):
        super(Ghost_Particles, self).__init__()
        self.n_ghosts = model_config.training.n_ghosts
        self.n_frames = model_config.simulation.n_frames
        self.n_dataset = model_config.training.n_runs
        self.vnorm = vnorm
        # self.model_siren = model_siren
        self.device = device

        self.ghost_pos = nn.Parameter(torch.rand((self.n_dataset, self.n_frames, self.n_ghosts, 2), device=device, requires_grad=True))
        if model_config.graph_model.particle_model_name == 'PDE_B':
            # self.ghost_dpos = nn.Parameter(torch.zeros((self.n_dataset, self.n_frames, self.n_ghosts, 2), device=device, requires_grad=True))
            self.boids = True
        else:
            self.boids = False
        self.N1 = torch.arange(n_particles,n_particles+self.n_ghosts, device=device, requires_grad=False)
        self.V1 = torch.zeros((self.n_ghosts,2), device=device, requires_grad=False)
        self.T1 = torch.zeros(self.n_ghosts, device=device, requires_grad=False)
        self.H1 = torch.zeros((self.n_ghosts,2), device=device, requires_grad=False)
        self.A1 = torch.zeros(self.n_ghosts, device=device, requires_grad=False)

        embedding_index = int(n_particles)
        embedding_index = np.arange(embedding_index)
        embedding_index = np.random.permutation(embedding_index)
        self.embedding_index = embedding_index[:self.n_ghosts]

    def get_pos (self, dataset_id, frame, bc_pos):

        out = torch.concatenate((self.N1[:,None], bc_pos(self.ghost_pos[dataset_id, frame:frame+1,:,:].squeeze()),self.V1,self.T1[:,None],self.H1,self.A1[:,None]), 1)

        return out

    
    
    

