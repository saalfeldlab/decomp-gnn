# SIREN network
# Code adapted from the following GitHub repository:
# https://github.com/vsitzmann/siren?tab=readme-ov-file
import os

import numpy as np
import torch
import torch.nn as nn

# from ParticleGraph.generators.utils import get_time_series
import matplotlib
from matplotlib import pyplot as plt
from tifffile import imread, imsave
from tqdm import trange



class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class Siren_Network(nn.Module):
    def __init__(self, image_width, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., device='cuda:0'):
        super().__init__()

        self.device = device 
        self.image_width = image_width

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)

        self.net = self.net.to(device)

    @property
    def values(self):
        # Call forward method
        output, coords = self.__call__()
        return output.squeeze().reshape(self.image_width, self.image_width)
    
    def coordinate_grid(self, n_points):
        coords = np.linspace(0, 1, n_points, endpoint=False)
        xy_grid = np.stack(np.meshgrid(coords, coords), -1)
        xy_grid = torch.tensor(xy_grid).unsqueeze(0).permute(0, 3, 1, 2).float().contiguous().to(self.device)
        return xy_grid
    
    def get_mgrid(self, sidelen, dim=2):
        '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
        sidelen: int
        dim: int'''
        tensors = tuple(dim * [torch.linspace(0, 1, steps=sidelen)])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = mgrid.reshape(-1, dim)
        return mgrid

    def forward(self, coords=None, time=None):
        if coords is None:
            coords = self.get_mgrid(self.image_width, dim=2).to(self.device)
            if time != None:
               coords = torch.cat((coords, time * torch.ones_like(coords[:, 0:1])), 1)

        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

if __name__ == '__main__':

    device = 'cuda:0'
    try:
        matplotlib.use("Qt5Agg")
    except:
        pass

    model_siren = Siren_Network(image_width=256, in_features=2, out_features=1, hidden_features=256, hidden_layers=3, outermost_linear=True)
    model_siren = model_siren.to(device=device)
    optimizer = torch.optim.Adam(lr=1e-4, params=model_siren.parameters())

    i0 = imread('pics_boat.tif')

    y = torch.tensor(i0, dtype=torch.float32, device=device)
    y = y.flatten()
    y = y[:,None]

    coords = get_mgrid(256, dim=2)
    coords = coords.to('cuda:0')

    print(coords.device, y.device)


    for epoch in trange(10000):
        optimizer.zero_grad()

        x = model_siren()**2

        loss = (x - y).norm(2)

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss {loss.item()}")
            pred = model_siren()**2
            pred = torch.reshape(pred, (256, 256))
            fig = plt.figure(figsize=(8, 8))
            plt.imshow(pred.detach().cpu().numpy())
            # plt.scatter(y.detach().cpu().numpy(),x.detach().cpu().numpy(),c='k',s=1)
            plt.savefig(f"tmp/output_{epoch}.png")
