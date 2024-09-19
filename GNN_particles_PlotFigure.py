import os

import torch
import matplotlib

from ParticleGraph.plotting import get_figures

os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/x86_64-linux'

# matplotlib.use("Qt5Agg")


if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(' ')
    print(f'device {device}')
    print(' ')

    try:
        matplotlib.use("Qt5Agg")
    except Exception as _:
        pass

    f_list = ['supp15']
    for f in f_list:
        config_list,epoch_list = get_figures(f, device=device)

