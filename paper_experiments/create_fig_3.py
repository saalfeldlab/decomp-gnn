import os

import torch
import matplotlib

from ParticleGraph.plotting import get_figures


if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(' ')
    print(f'device {device}')
    print(' ')

    f_list = ['3']
    for f in f_list:
        config_list,epoch_list = get_figures(f, device=device)

