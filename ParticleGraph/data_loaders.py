"""
A collection of functions for loading data from various sources.
"""
import os
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Literal

import astropy.units as u
import h5py
import numpy as np
import pandas as pd
import torch
from astropy.units import Unit
from scipy.interpolate import CubicSpline, interp1d, make_interp_spline
from tqdm import trange

from ParticleGraph.TimeSeries import TimeSeries
from ParticleGraph.utils import *


def skip_to(file, start_line):
    with open(file) as f:
        pos = 0
        cur_line = f.readline()
        while cur_line != start_line:
            pos += 1
            cur_line = f.readline()

        return pos + 1


def convert_data(data, device, config, n_particle_types, n_frames):
    x_list = []
    y_list = []

    for it in trange(n_frames - 1):
        for n in range(n_particle_types):
            # if (n==9):
            #     p=1
            x = data[n][it, 1].clone().detach()
            y = data[n][it, 2].clone().detach()
            z = data[n][it, 3].clone().detach()
            vx = data[n][it, 4].clone().detach()
            vy = data[n][it, 5].clone().detach()
            vz = data[n][it, 6].clone().detach()

            tmp = torch.stack(
                [torch.tensor(n), x, y, vx, vy, torch.tensor(n), torch.tensor(0), torch.tensor(0), torch.tensor(0)])
            if n == 0:
                object_data = tmp[None, :]
            else:
                object_data = torch.cat((object_data, tmp[None, :]), 0)

            ax = data[n][it + 1, 4] - data[n][it, 4]
            ay = data[n][it + 1, 5] - data[n][it, 5]
            tmp = torch.stack([ax, ay]) / config.simulation.delta_t
            if n == 0:
                acc_data = tmp[None, :]
            else:
                acc_data = torch.cat((acc_data, tmp[None, :]), 0)

        x_list.append(object_data.to(device))
        y_list.append(acc_data.to(device))

    return x_list, y_list


def load_solar_system(config, device=None, visualize=False, folder=None, step=1000):
    # create output folder, empty it if bErase=True, copy files into it
    dataset_name = config.data_folder_name
    simulation_config = config.simulation
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    n_step = simulation_config.n_frames + 3
    n_frames = simulation_config.n_frames
    # Start = 1980 - 03 - 06
    # Stop = 2013 - 03 - 06
    # Step = 4(hours)

    object_list = ['sun', 'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto', 'io',
                   'europa',
                   'ganymede', 'callisto', 'mimas', 'enceladus', 'tethys', 'dione', 'rhea', 'titan', 'hyperion', 'moon',
                   'phobos', 'deimos', 'charon']

    # matplotlib.use("Qt5Agg")
    fig = plt.figure(figsize=(12, 12))

    all_data = []

    for id, object in enumerate(object_list):

        print(f'object: {object}')
        filename = os.path.join(dataset_name, f'{object}.txt')

        df = skip_to(filename, "$$SOE\n")
        data = pd.read_csv(filename, header=None, skiprows=df, sep='', nrows=n_step)
        tmp_x = data.iloc[:, 4:5].values
        tmp_y = data.iloc[:, 5:6].values
        tmp_z = data.iloc[:, 6:7].values
        tmp_vx = data.iloc[:, 7:8].values
        tmp_vy = data.iloc[:, 8:9].values
        tmp_vz = data.iloc[:, 9:10].values
        tmp_x = tmp_x[:, 0][:-1]
        tmp_y = tmp_y[:, 0][:-1]
        tmp_z = tmp_z[:, 0][:-1]
        tmp_vx = tmp_vx[:, 0][:-1]
        tmp_vy = tmp_vy[:, 0][:-1]
        tmp_vz = tmp_vz[:, 0][:-1]
        # convert string to float
        x = torch.ones((n_step - 1))
        y = torch.ones((n_step - 1))
        z = torch.ones((n_step - 1))
        vx = torch.ones((n_step - 1))
        vy = torch.ones((n_step - 1))
        vz = torch.ones((n_step - 1))
        for it in range(n_step - 1):
            x[it] = torch.tensor(float(tmp_x[it][0:-1]))
            y[it] = torch.tensor(float(tmp_y[it][0:-1]))
            z[it] = torch.tensor(float(tmp_z[it][0:-1]))
            vx[it] = torch.tensor(float(tmp_vx[it][0:-1]))
            vy[it] = torch.tensor(float(tmp_vy[it][0:-1]))
            vz[it] = torch.tensor(float(tmp_vz[it][0:-1]))

        object_data = torch.cat((torch.ones_like(x[:, None]) * id, x[:, None], y[:, None], z[:, None], vx[:, None],
                                 vy[:, None], vz[:, None], torch.ones_like(x[:, None]) * id,
                                 torch.zeros_like(x[:, None]), torch.zeros_like(x[:, None]),
                                 torch.zeros_like(x[:, None])), 1)

        all_data.append(object_data)

        plt.plot(to_numpy(y), to_numpy(x))
        plt.text(to_numpy(y[-1]), to_numpy(x[-1]), object, fontsize=6)

    x_list, y_list = convert_data(all_data, device, config, n_particle_types, n_frames + 1)

    dataset_name = config.dataset

    if visualize:
        for it in trange(n_frames - 1):
            if it % step == 0:
                fig = plt.figure(figsize=(12, 12))
                for id, object in enumerate(object_list):
                    plt.scatter(to_numpy(x_list[it][id, 1]), to_numpy(x_list[it][id, 2]), s=20)
                    if id < 10:
                        plt.text(to_numpy(x_list[it][id, 1]), to_numpy(x_list[it][id, 2]), object, fontsize=6)
                    if id == 9:
                        plt.arrow(to_numpy(x_list[it][id, 1]), to_numpy(x_list[it][id, 2]),
                                  to_numpy(y_list[it][id, 0]) * 1E14, to_numpy(y_list[it][id, 1]) * 1E14,
                                  head_width=0.5, head_length=0.7, fc='k', ec='k')
                plt.xlim([-0.5E10, 0.5E10])
                plt.ylim([-0.5E10, 0.5E10])
                plt.tight_layout()
                plt.savefig(f"graphs_data/graphs_{dataset_name}/generated_data/Fig_{it}.jpg", dpi=170.7)
                plt.close()

    for run in range(2):
        torch.save(x_list, f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt')
        torch.save(y_list, f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt')

def ensure_local_path_exists(path):
    """
    Ensure that the local path exists. If it doesn't, create the directory structure.

    :param path: The path to be checked and created if necessary.
    :return: The absolute path of the created directory.
    """

    os.makedirs(path, exist_ok=True)
    return os.path.join(os.getcwd(), path)


@dataclass
class CsvDescriptor:
    """A class to describe the location of data in a dataset as a column of a CSV file."""
    filename: str
    column_name: str
    type: np.dtype
    unit: Unit


def load_csv_from_descriptors(
        column_descriptors: Dict[str, CsvDescriptor],
        **kwargs
) -> pd.DataFrame:
    """
    Load data from a CSV file based on a set of column descriptors.

    :param column_descriptors: A dictionary mapping field names to CsvDescriptors.
    :param kwargs: Additional keyword arguments to pass to pd.read_csv.
    :return: A pandas DataFrame containing the loaded data.
    """
    different_files = set(descriptor.filename for descriptor in column_descriptors.values())
    columns = []

    for file in different_files:
        dtypes = {descriptor.column_name: descriptor.type for descriptor in column_descriptors.values()
                  if descriptor.filename == file}
        print(f"Loading data from '{file}':")
        for column_name, dtype in dtypes.items():
            print(f"  - column {column_name} as {dtype}")
        columns.append(pd.read_csv(file, dtype=dtypes, usecols=list(dtypes.keys()), **kwargs))

    data = pd.concat(columns, axis='columns')
    data.rename(columns={descriptor.column_name: name for name, descriptor in column_descriptors.items()}, inplace=True)

    return data
