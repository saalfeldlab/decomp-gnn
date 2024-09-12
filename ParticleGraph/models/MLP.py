import torch.nn as nn
import torch.nn.functional as F

from ParticleGraph.generators.utils import get_time_series
import matplotlib
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import os


class MLP(nn.Module):

    def __init__(self, input_size=None, output_size=None, nlayers=None, hidden_size=None, device=None, activation=None, initialisation=None):

        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size, device=device))
        if nlayers > 2:
            for i in range(1, nlayers - 1):
                layer = nn.Linear(hidden_size, hidden_size, device=device)
                nn.init.normal_(layer.weight, std=0.1)
                nn.init.zeros_(layer.bias)
                self.layers.append(layer)
        layer = nn.Linear(hidden_size, output_size, device=device)

        if initialisation == 'zeros':
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        else :
            nn.init.normal_(layer.weight, std=0.1)
            nn.init.zeros_(layer.bias)
        self.layers.append(layer)

        if activation=='tanh':
            self.activation = F.tanh
        else:
            self.activation = F.relu

    def forward(self, x):
        for l in range(len(self.layers) - 1):
            x = self.layers[l](x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x

if __name__ == '__main__':

    device = 'cuda:0'
    matplotlib.use("Qt5Agg")



    current_path = os.path.dirname(os.path.realpath(__file__))

    time_series = torch.load('../../../graphs_data/graphs_boids_16_256_division_model_2_mass_coeff/x_list_0.pt', map_location=device)

    mass_list = get_time_series(time_series,500,'mass')
    mass_list = torch.tensor(mass_list[0:2000], device=device) / 500
    age_list = get_time_series(time_series,500,'age') / 250
    age_list = torch.tensor(age_list[0:2000], device=device)

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(age_list.detach().cpu().numpy(), mass_list.detach().cpu().numpy())

    fig = plt.figure(figsize=(8, 8))
    plt.plot(mass_list.detach().cpu().numpy())


    model = MLP(input_size=1, output_size=1, nlayers=5, hidden_size=64, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1E-2)
    model.train()

    for epoch in range(10000):
        optimizer.zero_grad()

        sample = np.random.randint(0, len(mass_list), 10)

        MLP_pred = model(age_list[sample,None])
        y = mass_list[sample,None]

        loss = (MLP_pred - y).norm(2)

        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss {loss.item()}")

    pred = model(age_list[:,None])
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(age_list.detach().cpu().numpy(), pred.detach().cpu().numpy(),alpha=0.1)
    plt.scatter(age_list.detach().cpu().numpy(), mass_list.detach().cpu().numpy(),alpha=0.1)



