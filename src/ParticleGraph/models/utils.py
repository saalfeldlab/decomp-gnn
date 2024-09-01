import matplotlib.pyplot as plt
import torch
import umap
from matplotlib.ticker import FormatStrFormatter
from ParticleGraph.models import Interaction_Particle, Interaction_Particle_Field, Signal_Propagation, Mesh_Laplacian, Mesh_RPS
from ParticleGraph.utils import *

import matplotlib as mpl
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import warnings
import numpy as np
import time
import tqdm
import seaborn as sns
from tifffile import imsave

def linear_model(x, a, b):
    return a * x + b

def get_embedding(model_a=None, dataset_number = 0):
    embedding = []
    embedding.append(model_a[dataset_number])
    embedding = to_numpy(torch.stack(embedding).squeeze())

    return embedding

def get_embedding_time_series(model=None, dataset_number=None, cell_id=None, n_particles=None, n_frames=None, has_cell_division=None):
    embedding = []
    embedding.append(model.a[dataset_number])
    embedding = to_numpy(torch.stack(embedding).squeeze())

    indexes = np.arange(n_frames) * n_particles + cell_id

    return embedding[indexes]

def get_type_time_series(new_labels=None, dataset_number=None, cell_id=None, n_particles=None, n_frames=None, has_cell_division=None):

    indexes = np.arange(n_frames) * n_particles + cell_id

    return new_labels[indexes]

def get_in_features(rr, embedding_, config_model, max_radius):
    match config_model:
        case 'PDE_A' | 'PDE_Cell_A':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding_), dim=1)
        case 'PDE_ParticleField_A':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding_), dim=1)
        case 'PDE_A_bis':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding_, embedding_), dim=1)
        case 'PDE_B' | 'PDE_Cell_B':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
        case 'PDE_ParticleField_B':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
        case 'PDE_GS':
            in_features = torch.cat(
                (rr[:, None] / max_radius, 0 * rr[:, None], rr[:, None] / max_radius, 10 ** embedding_), dim=1)
        case 'PDE_G':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None],
                                     0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
        case 'PDE_E':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding_, embedding_), dim=1)
        case 'PDE_N' | 'PDE_N2':
            in_features = torch.cat((rr[:, None], embedding_), dim=1)

    return in_features

def plot_training_signal(config, dataset_name, model, adjacency, ynorm, log_dir, epoch, N, index_particles, n_particles, n_particle_types, type_list, cmap, device):

    fig = plt.figure(figsize=(8, 8))
    embedding = get_embedding(model.a, 1)
    for n in range(n_particle_types):
        plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], s=20)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding/{dataset_name}_{epoch}_{N}.tif", dpi=87)
    plt.close()

    fig = plt.figure(figsize=(8, 8))
    rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
    for n in range(n_particles):
        if 'PDE_N2' in config.graph_model.signal_model_name:
            in_features =rr[:, None]
        else:
            embedding_ = model.a[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
            in_features = torch.cat((rr[:, None], embedding_), dim=1)
        with torch.no_grad():
            func = model.lin_phi(in_features.float())
        if (n % 2 == 0):
            plt.plot(to_numpy(rr), to_numpy(func),2, color=cmap.color(to_numpy(type_list)[n].astype(int)), linewidth=2, alpha=0.25)
    plt.savefig(f"./{log_dir}/tmp_training/function/{dataset_name}_{epoch}_{N}.tif", dpi=87)

    i, j = torch.triu_indices(n_particles, n_particles, requires_grad=False, device=device)
    if 'PDE_N2' in config.graph_model.signal_model_name:
        A = model.W.clone().detach()
        A[i,i] = 0
    elif 'asymmetric' in config.simulation.adjacency_matrix:
        A = model.vals
    else:
        A = torch.zeros(n_particles, n_particles, device=device, requires_grad=False, dtype=torch.float32)
        A[i, j] = model.vals
        A.T[i, j] = model.vals
    fig = plt.figure(figsize=(8, 8))
    ax = sns.heatmap(to_numpy(A),center=0,square=True,cmap='bwr',cbar_kws={'fraction':0.046}, vmin=-0.01, vmax=0.01)
    plt.title('Random connectivity matrix',fontsize=12);
    plt.xticks([0,n_particles-1],[1,n_particles],fontsize=8)
    plt.yticks([0,n_particles-1],[1,n_particles],fontsize=8)
    plt.imshow(to_numpy(A), cmap='viridis')
    plt.savefig(f"./{log_dir}/tmp_training/field/{dataset_name}_{epoch}_{N}.tif", dpi=87)
    plt.close()

    imsave(f"./{log_dir}/tmp_training/field/adjacency_{dataset_name}__{epoch}_{N}.tif", to_numpy(A))

def plot_training_particle_field(config, has_siren, has_siren_time, model_f, dataset_name, n_frames, model_name, log_dir, epoch, N, x, x_mesh, model_field, index_particles, n_particles, n_particle_types, model, n_nodes, n_node_types, index_nodes, dataset_num, ynorm, cmap, axis, device):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    max_radius = simulation_config.max_radius

    n_nodes = simulation_config.n_nodes
    n_nodes_per_axis = int(np.sqrt(n_nodes))

    fig = plt.figure(figsize=(12, 12))
    if axis:
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
        # plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
    else:
        plt.axis('off')
    embedding = get_embedding(model.a, dataset_num)
    if n_particle_types > 1000:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=to_numpy(x[:, 5]) / n_particles, s=5, cmap='viridis')
    else:
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=200)  #

    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding/{model_name}_{dataset_name}_embedding_{epoch}_{N}.tif", dpi=170.7)
    plt.close()

    fig = plt.figure(figsize=(12, 12))
    if axis:
        ax = fig.add_subplot(1, 1, 1)
        # ax.xaxis.get_major_formatter()._usetex = False
        # ax.yaxis.get_major_formatter()._usetex = False
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        # plt.xlabel(r'$d_{ij}$', fontsize=64)
        # plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlim([0, simulation_config.max_radius])
        # plt.ylim([-0.15, 0.15])
        # plt.ylim([-0.04, 0.03])
        # plt.ylim([-0.1, 0.1])
        plt.tight_layout()

    match model_config.particle_model_name:
        case 'PDE_ParticleField_A':
            rr = torch.tensor(np.linspace(0, simulation_config.max_radius, 200)).to(device)
        case 'PDE_ParticleField_B':
            rr = torch.tensor(np.linspace(-max_radius, max_radius, 200)).to(device)
    for n in range(n_particles):
        embedding_ = model.a[dataset_num, n, :] * torch.ones((200, model_config.embedding_dim), device=device)
        match model_config.particle_model_name:
            case 'PDE_ParticleField_A':
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None], rr[:, None] / max_radius, embedding_), dim=1)
            case 'PDE_ParticleField_B':
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
        with torch.no_grad():
            func = model.lin_edge(in_features.float())
        func = func[:, 0]
        if n % 5 == 0:
            plt.plot(to_numpy(rr),
                     to_numpy(func * ynorm),
                     linewidth=8,
                     color=cmap.color(to_numpy(x[n, 5]).astype(int)), alpha=0.25)
    match model_config.particle_model_name:
        case 'PDE_ParticleField_A':
            plt.ylim([-0.04, 0.03])
        case 'PDE_ParticleField_B':
            plt.ylim([-1E-4, 1E-4])

    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/function/{model_name}_{dataset_name}_function_{epoch}_{N}.tif", dpi=170.7)
    plt.close()

    fig = plt.figure(figsize=(12, 12))
    if has_siren:
        if has_siren_time:
            frame_list = [n_frames//4, 2*n_frames//4, 2*n_frames//4, n_frames-1]
        else:
            frame_list = [0]

        for frame in frame_list:

            if has_siren_time:
                with torch.no_grad():
                    tmp = model_f(time=frame / n_frames) ** 2
            else:
                with torch.no_grad():
                    tmp = model_f() ** 2
            tmp = torch.reshape(tmp, (n_nodes_per_axis, n_nodes_per_axis))
            tmp = to_numpy(torch.sqrt(tmp))
            if has_siren_time:
                tmp= np.rot90(tmp,k=1)
            fig_ = plt.figure(figsize=(12, 12))
            axf = fig_.add_subplot(1, 1, 1)
            plt.imshow(tmp, cmap='grey', vmin=0, vmax=2)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/field/{model_name}_{epoch}_{N}_{frame}.tif", dpi=170.7)
            plt.close()

    else:
        im = to_numpy(model_field[dataset_num])
        im = np.reshape(im, (n_nodes_per_axis, n_nodes_per_axis))
        plt.imshow(im)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/field/{model_name}_{dataset_name}_field_{epoch}_{N}.tif", dpi=87)
        plt.close()

    # im = np.flipud(im)
    # io.imsave(f"./{log_dir}/tmp_training/field_pic_{dataset_name}_{epoch}_{N}.tif", im)

def plot_training (config, dataset_name, log_dir, epoch, N, x, index_particles, n_particles, n_particle_types, model, n_nodes, n_node_types, index_nodes, dataset_num, ynorm, cmap, axis, device):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model
    do_tracking = train_config.do_tracking
    matplotlib.rcParams['savefig.pad_inches'] = 0

    if model_config.mesh_model_name == 'WaveMesh':
        fig = plt.figure(figsize=(8, 8))
        rr = torch.tensor(np.linspace(-150, 150, 200)).to(device)
        popt_list = []
        for n in range(n_nodes):
            embedding_ = model.a[dataset_num, n, :] * torch.ones((200, 2), device=device)
            in_features = torch.cat((rr[:, None], embedding_), dim=1)
            h = model.lin_phi(in_features.float())
            h = h[:, 0]
            popt, pcov = curve_fit(linear_model, to_numpy(rr.squeeze()), to_numpy(h.squeeze()))
            popt_list.append(popt)
        t = np.array(popt_list)
        t = t[:, 0]
        plt.close()

        fig = plt.figure(figsize=(8, 8))
        embedding = get_embedding(model.a, 1)
        plt.scatter(embedding[:, 0], embedding[:, 1], c=t[:, None], s=20, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/embedding/{dataset_name}_{epoch}_{N}.tif",dpi=87)
        plt.close()

        fig = plt.figure(figsize=(8, 8))
        t = np.reshape(t, (100, 100))
        t = np.flipud(t)
        plt.imshow(t, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/field/mesh_map_{dataset_name}_{epoch}_{N}.tif",
                    dpi=87)
        plt.close()
    elif model_config.mesh_model_name == 'RD_RPS_Mesh':
        fig = plt.figure(figsize=(8, 8))
        plt.savefig(f"./{log_dir}/tmp_training/field/mesh_map_{dataset_name}_{epoch}_{N}.tif",
                    dpi=87)
    else:
        fig = plt.figure(figsize=(8, 8))
        if do_tracking:
            embedding = to_numpy(model.a)
        else:
            embedding = get_embedding(model.a, 1)
        for n in range(n_particle_types):
            plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], s=20)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/embedding/{dataset_name}_{epoch}_{N}.tif",dpi=87)
        plt.close()

        match model_config.particle_model_name:
            case 'PDE_GS':
                fig = plt.figure(figsize=(8, 4))
                ax = fig.add_subplot(1, 2, 1)
                rr = torch.tensor(np.logspace(7, 9, 1000)).to(device)
                for n in range(n_particles):
                    if do_tracking:
                        embedding_ = model.a[n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
                    else:
                        embedding_ = model.a[1, n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
                    in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                             rr[:, None] / simulation_config.max_radius, 10 ** embedding_), dim=1)
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float())
                    func = func[:, 0]
                    plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),
                             color=cmap.color(to_numpy(x[n, 5]).astype(int)), linewidth=1)
                plt.xlabel('Distance [a.u]', fontsize=14)
                plt.ylabel('MLP [a.u]', fontsize=14)
                plt.xscale('log')
                plt.yscale('log')
                plt.tight_layout()
                ax = fig.add_subplot(1, 2, 2)
                plt.scatter(np.log(np.abs(to_numpy(y_batch[:, 0]))), np.log(np.abs(to_numpy(pred[:, 0]))), c='k', s=1,
                            alpha=0.15)
                plt.scatter(np.log(np.abs(to_numpy(y_batch[:, 1]))), np.log(np.abs(to_numpy(pred[:, 1]))), c='k', s=1,
                            alpha=0.15)
                plt.xlim([-10, 4])
                plt.ylim([-10, 4])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/function/func_{dataset_name}_{epoch}_{N}.tif", dpi=87)
                plt.close()

            case 'PDE_B' | 'PDE_ParticleField_B':
                max_radius = 0.04
                fig = plt.figure(figsize=(12, 12))
                # plt.rcParams['text.usetex'] = True
                # rc('font', **{'family': 'serif', 'serif': ['Palatino']})
                ax = fig.add_subplot(1,1,1)
                rr = torch.tensor(np.linspace(-max_radius, max_radius, 1000)).to(device)
                func_list = []
                for n in range(n_particles):
                    if do_tracking:
                        embedding_ = model.a[n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
                    else:
                        embedding_ = model.a[1, n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
                    in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                             torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                             0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float())
                    func = func[:, 0]
                    func_list.append(func)
                    if n % 5 == 0:
                        plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),
                                 color=cmap.color(int(n // (n_particles / n_particle_types))), linewidth=2)
                if not(do_tracking):
                    plt.ylim([-1E-4, 1E-4])
                plt.xlim([-max_radius, max_radius])
                # plt.xlabel(r'$x_j-x_i$', fontsize=64)
                # plt.ylabel(r'$f_{ij}$', fontsize=64)
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                ax.yaxis.set_major_locator(plt.MaxNLocator(5))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                fmt = lambda x, pos: '{:.1f}e-5'.format((x) * 1e5, pos)
                ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                plt.xticks(fontsize=32.0)
                plt.yticks(fontsize=32.0)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/function/{dataset_name}_function_{epoch}_{N}.tif",dpi=170.7)
                plt.close()

            case 'PDE_G':
                fig = plt.figure(figsize=(12, 12))
                if axis:
                    ax = fig.add_subplot(1, 1, 1)
                    # ax.xaxis.get_major_formatter()._usetex = False
                    # ax.yaxis.get_major_formatter()._usetex = False
                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                    plt.xticks(fontsize=32)
                    plt.yticks(fontsize=32)
                    plt.xlim([0, simulation_config.max_radius])
                    # plt.ylim([-0.15, 0.15])
                    # plt.ylim([-0.04, 0.03])
                    # plt.ylim([-0.1, 0.1])
                    plt.tight_layout()
                rr = torch.tensor(np.linspace(simulation_config.min_radius, simulation_config.max_radius, 1000)).to(device)
                for n in range(n_particles):
                    embedding_ = model.a[dataset_num, n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
                    in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                         rr[:, None] / simulation_config.max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float())
                    func = func[:, 0]
                    plt.plot(to_numpy(rr),
                             to_numpy(func*ynorm),
                             linewidth=8,
                             color=cmap.color(to_numpy(x[n, 5]).astype(int)), alpha=0.25)
                plt.xlim([0, 0.02])
                plt.ylim([0, 0.5E6])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/function/{dataset_name}_function_{epoch}_{N}.tif", dpi=87)
                plt.close()

            case 'PDE_A'| 'PDE_A_bis' | 'PDE_ParticleField_A' | 'PDE_E':
                fig = plt.figure(figsize=(12, 12))
                if axis:
                    ax = fig.add_subplot(1, 1, 1)
                    # ax.xaxis.get_major_formatter()._usetex = False
                    # ax.yaxis.get_major_formatter()._usetex = False
                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                    # plt.xlabel(r'$d_{ij}$', fontsize=64)
                    # plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
                    plt.xticks(fontsize=32)
                    plt.yticks(fontsize=32)
                    plt.xlim([0, simulation_config.max_radius])
                    # plt.ylim([-0.15, 0.15])
                    # plt.ylim([-0.04, 0.03])
                    # plt.ylim([-0.1, 0.1])
                    plt.tight_layout()
                rr = torch.tensor(np.linspace(0, simulation_config.max_radius, 200)).to(device)
                for n in range(n_particles):
                    if do_tracking:
                        embedding_ = model.a[n, :] * torch.ones((200, model_config.embedding_dim), device=device)
                    else:
                        embedding_ = model.a[1, n, :] * torch.ones((200, model_config.embedding_dim), device=device)
                    if (model_config.particle_model_name == 'PDE_A'):
                        in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                                 rr[:, None] / simulation_config.max_radius, embedding_), dim=1)
                    elif (model_config.particle_model_name == 'PDE_A_bis'):
                        in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                                 rr[:, None] / simulation_config.max_radius, embedding_, embedding_), dim=1)
                    elif (model_config.particle_model_name == 'PDE_B'):
                        in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                                 rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                                 0 * rr[:, None],
                                                 0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                    elif model_config.particle_model_name == 'PDE_E':
                        in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                                 rr[:, None] / simulation_config.max_radius, embedding_, embedding_), dim=1)
                    else:
                        in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                                 rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                                 0 * rr[:, None],
                                                 0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                    with torch.no_grad():
                        func = model.lin_edge(in_features.float())
                    func = func[:, 0]
                    if n % 5 == 0:
                        plt.plot(to_numpy(rr),
                                 to_numpy(func*ynorm),
                                 linewidth=2,
                                 color=cmap.color(to_numpy(x[n, 5]).astype(int)), alpha=0.25)
                if not (do_tracking):
                    plt.ylim(config.plotting.ylim)
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/function/{dataset_name}_function_{epoch}_{N}.tif", dpi=87)
                plt.close()

def plot_training_cell_tracking(config, id_list, dataset_name, log_dir, epoch, N, model, n_particle_types, type_list, ynorm, cmap, device):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    fig = plt.figure(figsize=(8, 8))
    for k in range(1,len(type_list),len(type_list)//40):
        for n in range(n_particle_types):
            pos =torch.argwhere(type_list[k] == n)
            if len(pos) > 0:
                if model.use_hot_encoding:
                    embedding = to_numpy(model.cc + torch.matmul(model.a[to_numpy(id_list[k][pos]).astype(int), :], model.basis).squeeze())
                else:
                    embedding = to_numpy(model.a[to_numpy(id_list[k][pos]).astype(int)].squeeze())
                plt.scatter(embedding[:, 0], embedding[:, 1], s=1, color=cmap.color(n), alpha=0.5)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding/{dataset_name}_{epoch}_{N}.tif", dpi=87)
    plt.close()


    max_radius = 0.04
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1,1,1)
    rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)

    for k in range(1,len(type_list), 10):
        for n in range(1,len(type_list[k]),10):
                if model.use_hot_encoding:
                    embedding_ = model.cc + torch.matmul(model.a[to_numpy(id_list[k][n]).astype(int), :], model.basis).squeeze()
                else:
                    embedding_ = model.a[to_numpy(id_list[k][n]).astype(int)]
                embedding_ = embedding_ * torch.ones((1000, model_config.embedding_dim), device=device)

                match model_config.particle_model_name:
                    case 'PDE_Cell_A':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 rr[:, None] / max_radius, embedding_), dim=1)
                    case 'PDE_Cell_A_area':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 rr[:, None] / max_radius, torch.ones_like(rr[:, None])*0.1, torch.ones_like(rr[:, None])*0.4, embedding_, embedding_), dim=1)
                    case 'PDE_Cell_B':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                                 0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                    case 'PDE_Cell_B_area':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                                 0 * rr[:, None], 0 * rr[:, None], torch.ones_like(rr[:, None])*0.001, torch.ones_like(rr[:, None])*0.001, embedding_, embedding_), dim=1)

                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                func = func[:, 0]
                type = to_numpy(type_list[k][n])
                plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),
                             color=cmap.color(int(type)), linewidth=2,alpha=0.1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fmt = lambda x, pos: '{:.1f}e-5'.format((x) * 1e5, pos)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    plt.xticks(fontsize=32.0)
    plt.yticks(fontsize=32.0)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/function/{dataset_name}_{epoch}_{N}.tif",dpi=87)
    plt.close()

def plot_training_cell(config, dataset_name, log_dir, epoch, N, model, n_particle_types, type_list, ynorm, cmap, device):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model
    matplotlib.rcParams['savefig.pad_inches'] = 0

    embedding = get_embedding(model.a, 1)
    fig = plt.figure(figsize=(8, 8))
    for n in range(n_particle_types):
        pos =torch.argwhere(type_list == n)
        pos = to_numpy(pos)
        if len(pos) > 0:
            plt.scatter(embedding[pos, 0], embedding[pos, 1], s=10, alpha=1)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding/{dataset_name}_{epoch}_{N}.tif", dpi=87)
    plt.close()

    max_radius = 0.04
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1,1,1)
    rr = torch.tensor(np.linspace(-max_radius, max_radius, 1000)).to(device)
    func_list = []

    if len(type_list) > 10000:
        step = len(type_list) // 1000
    else:
        step = 5

    for n in range(1,len(type_list),step):
        embedding_ = model.a[1, n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
        match model_config.particle_model_name:
            case 'PDE_Cell_A':
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         rr[:, None] / max_radius, embedding_), dim=1)
            case 'PDE_Cell_A_area':
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         rr[:, None] / max_radius, torch.ones_like(rr[:, None])*0.1, torch.ones_like(rr[:, None])*0.4, embedding_, embedding_), dim=1)
            case 'PDE_Cell_B':
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
            case 'PDE_Cell_B_area':
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], torch.ones_like(rr[:, None])*0.001, torch.ones_like(rr[:, None])*0.001, embedding_, embedding_), dim=1)

        with torch.no_grad():
            func = model.lin_edge(in_features.float())
        func = func[:, 0]
        func_list.append(func)
        plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color( int(type_list[n])   ), linewidth=2)
    plt.xlim([-max_radius, max_radius])
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fmt = lambda x, pos: '{:.1f}e-5'.format((x) * 1e5, pos)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    plt.xticks(fontsize=32.0)
    plt.yticks(fontsize=32.0)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/function/{dataset_name}_{epoch}_{N}.tif",dpi=87)
    plt.close()

def analyze_edge_function_tracking(rr=[], vizualize=False, config=None, model_MLP=[], model_a=None, n_particles=None, ynorm=None, indexes=None, type_list=None, cmap=None, dimension=2, embedding_type=0, device=None):

    model_config = config.graph_model
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius

    if rr==[]:
        if model_config.particle_model_name == 'PDE_G':
            rr = torch.tensor(np.linspace(0, max_radius * 1.3, 1000)).to(device)
        elif model_config.particle_model_name == 'PDE_GS':
            rr = torch.tensor(np.logspace(7, 9, 1000)).to(device)
        elif model_config.particle_model_name == 'PDE_E':
            rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
        else:
            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)

    if embedding_type == 1:
        n_list = indexes
    else:
        n_list = range(n_particles)

    func_list = []
    for n, k  in enumerate(n_list):
        embedding_ = model_a[int(k), :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        if config.graph_model.particle_model_name != '':
            config_model = config.graph_model.particle_model_name
        elif config.graph_model.signal_model_name != '':
            config_model = config.graph_model.signal_model_name
        elif config.graph_model.mesh_model_name != '':
            config_model = config.graph_model.mesh_model_name
        in_features = get_in_features(rr, embedding_, config_model, max_radius)
        with torch.no_grad():
            func = model_MLP(in_features.float())
        func = func[:, 0]
        func_list.append(func)
        if ((n % 5 == 0) | (config.graph_model.particle_model_name=='PDE_GS')) & vizualize:
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color(int(type_list[int(n)])), linewidth=2, alpha=0.25)
    func_list = torch.stack(func_list)
    coeff_norm = to_numpy(func_list)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if coeff_norm.shape[0] > 1000:
            new_index = np.random.permutation(coeff_norm.shape[0])
            new_index = new_index[0:min(1000, coeff_norm.shape[0])]
            trans = umap.UMAP(n_neighbors=500, n_components=2, transform_queue_size=0, random_state=config.training.seed).fit(coeff_norm[new_index])
            proj_interaction = trans.transform(coeff_norm)
        else:
            trans = umap.UMAP(n_neighbors=100, n_components=2, transform_queue_size=0).fit(coeff_norm)
            proj_interaction = trans.transform(coeff_norm)

    if vizualize:
        if config.graph_model.particle_model_name == 'PDE_GS':
            plt.xscale('log')
            plt.yscale('log')
        if config.graph_model.particle_model_name == 'PDE_G':
            plt.xscale('log')
            plt.yscale('log')
            plt.xlim([1E-3, 0.2])
        if config.graph_model.particle_model_name == 'PDE_E':
            plt.xlim([0, 0.05])
        plt.xlabel('Distance [a.u]', fontsize=12)
        plt.ylabel('MLP [a.u]', fontsize=12)

    return func_list, proj_interaction

def analyze_edge_function_state(rr=[], config=None, model=None, id_list=None, type_list=None, cmap=None, ynorm=None, visualize=False, device=None):

    max_radius = config.simulation.max_radius
    state_hot_encoding = config.training.state_hot_encoding

    if config.graph_model.particle_model_name != '':
        config_model = config.graph_model.particle_model_name
    elif config.graph_model.signal_model_name != '':
        config_model = config.graph_model.signal_model_name
    elif config.graph_model.mesh_model_name != '':
        config_model = config.graph_model.mesh_model_name

    func_list = []
    true_type_list = []
    short_model_a_list = []
    rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
    for k in range(1,len(type_list), 10):
        for n in range(1,len(type_list[k]),10):
                short_model_a_list.append(model.a[to_numpy(id_list[k][n]).astype(int)])
                if config.training.use_hot_encoding:
                    embedding_ = model.cc + torch.matmul(model.a[to_numpy(id_list[k][n]).astype(int), :], model.basis).squeeze()
                else:
                    embedding_ = model.a[to_numpy(id_list[k][n]).astype(int)]
                embedding_ = embedding_ * torch.ones((1000, config.simulation.dimension), device=device)

                match config_model:
                    case 'PDE_Cell_A':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 rr[:, None] / max_radius, embedding_), dim=1)
                    case 'PDE_Cell_A_area':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 rr[:, None] / max_radius, torch.ones_like(rr[:, None])*0.1, torch.ones_like(rr[:, None])*0.4, embedding_, embedding_), dim=1)
                    case 'PDE_Cell_B':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                                 0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                    case 'PDE_Cell_B_area':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                                 0 * rr[:, None], 0 * rr[:, None], torch.ones_like(rr[:, None])*0.001, torch.ones_like(rr[:, None])*0.001, embedding_, embedding_), dim=1)

                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                func = func[:, 0]
                func_list.append(func)
                true_type_list.append(type_list[k][n])
                if visualize:
                    plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),color=cmap.color(int(type_list[k][n])), linewidth=2, alpha=0.25)

    if visualize:
        plt.xlabel('Distance [a.u]')
        plt.ylabel('MLP [a.u]')
        plt.tight_layout()

    func_list = torch.stack(func_list)
    func_list_ = to_numpy(func_list)
    true_type_list = torch.stack(true_type_list)
    true_type_list = to_numpy(true_type_list)
    short_model_a_list = torch.stack(short_model_a_list)

    print('UMAP reduction ...')
    start_time = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        trans = umap.UMAP(n_neighbors=100, n_components=2, transform_queue_size=0).fit(func_list_)
    computation_time = time.time() - start_time
    print(f"UMAP computation time is {np.round(computation_time)} seconds.")

    proj_interaction = trans.transform(func_list_)
    proj_interaction = (proj_interaction - np.min(proj_interaction)) / (np.max(proj_interaction) - np.min(proj_interaction) + 1e-10)

    computation_time = time.time() - start_time
    print(f"dimension reduction computation time is {np.round(computation_time)} seconds.")

    return func_list, true_type_list, short_model_a_list, proj_interaction

def analyze_edge_function(rr=[], vizualize=False, config=None, model_MLP=[], model_a=None, n_nodes=0, dataset_number = 0, n_particles=None, ynorm=None, type_list=None, cmap=None, dimension=2, device=None):

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius

    if config.graph_model.particle_model_name != '':
        config_model = config.graph_model.particle_model_name
    elif config.graph_model.signal_model_name != '':
        config_model = config.graph_model.signal_model_name
    elif config.graph_model.mesh_model_name != '':
        config_model = config.graph_model.mesh_model_name

    if rr==[]:
        if config_model == 'PDE_G':
            rr = torch.tensor(np.linspace(0, max_radius * 1.3, 1000)).to(device)
        elif config_model == 'PDE_GS':
            rr = torch.tensor(np.logspace(7, 9, 1000)).to(device)
        elif config_model == 'PDE_E':
            rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
        elif 'PDE_N' in config_model:
            rr = torch.tensor(np.linspace(0, 0.9, 1000)).to(device)
        else:
            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)

    print('interaction functions ...')
    func_list = []
    for n in range(n_particles):
        if config.training.do_tracking:
            embedding_ = model_a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        else:
            embedding_ = model_a[dataset_number, n_nodes+n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)

        in_features = get_in_features(rr, embedding_, config_model, max_radius)
        with torch.no_grad():
            func = model_MLP(in_features.float())

        func = func[:, 0]
        func_list.append(func)
        if ((n % 5 == 0) | (config.graph_model.particle_model_name=='PDE_GS') | ('PDE_N' in config_model)) & vizualize:
            plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),2, color=cmap.color(type_list[n].astype(int)), linewidth=2, alpha=0.25)

    func_list = torch.stack(func_list)
    func_list_ = to_numpy(func_list)

    print('UMAP reduction ...')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if func_list_.shape[0] > 1000:
            new_index = np.random.permutation(func_list_.shape[0])
            new_index = new_index[0:min(1000, func_list_.shape[0])]
            trans = umap.UMAP(n_neighbors=500, n_components=2, transform_queue_size=0, random_state=config.training.seed).fit(func_list_[new_index])
            proj_interaction = trans.transform(func_list_)
        else:
            trans = umap.UMAP(n_neighbors=50, n_components=2, transform_queue_size=0).fit(func_list_)
            proj_interaction = trans.transform(func_list_)
    print('done ...')

    if vizualize:
        if config.graph_model.particle_model_name == 'PDE_GS':
            plt.xscale('log')
            plt.yscale('log')
        if config.graph_model.particle_model_name == 'PDE_G':
            plt.xscale('log')
            plt.yscale('log')
            plt.xlim([1E-3, 0.2])
        if config.graph_model.particle_model_name == 'PDE_E':
            plt.xlim([0, 0.05])
        plt.xlabel('Distance [a.u]')
        plt.ylabel('MLP [a.u]')
        plt.tight_layout()

    return func_list, proj_interaction

def choose_training_model(model_config, device):
    
    aggr_type = model_config.graph_model.aggr_type
    n_particle_types = model_config.simulation.n_particle_types
    n_particles = model_config.simulation.n_particles
    dimension = model_config.simulation.dimension
    do_tracking = model_config.training.do_tracking

    bc_pos, bc_dpos = choose_boundary_values(model_config.simulation.boundary)

    model=[]
    model_name = model_config.graph_model.particle_model_name
    match model_name:
        case 'PDE_ParticleField_A' | 'PDE_ParticleField_B':
            model = Interaction_Particle_Field(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos,
                                          dimension=dimension)
            model.edges = []
        case 'PDE_A' | 'PDE_A_bis' | 'PDE_B' | 'PDE_B_mass' | 'PDE_B_bis' | 'PDE_E' | 'PDE_G':
            model = Interaction_Particle(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
            model.edges = []
    model_name = model_config.graph_model.mesh_model_name
    match model_name:
        case 'DiffMesh':
            model = Mesh_Laplacian(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []
        case 'WaveMesh':
            model = Mesh_Laplacian(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []
        case 'RD_RPS_Mesh':
            model = Mesh_RPS(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []
        case 'RD_RPS_Mesh_bis':
            model = Mesh_RPS_bis(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []
    model_name = model_config.graph_model.signal_model_name
    match model_name:
        case 'PDE_N':
            model = Signal_Propagation(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []

    if model==[]:
        raise ValueError(f'Unknown model {model_name}')

    return model, bc_pos, bc_dpos

def constant_batch_size(batch_size):
    def get_batch_size(epoch):
        return batch_size

    return get_batch_size

def increasing_batch_size(batch_size):
    def get_batch_size(epoch):
        return 1 if epoch < 1 else batch_size

    return get_batch_size

def set_trainable_parameters(model, lr_embedding, lr):

    trainable_params = [param for _, param in model.named_parameters() if param.requires_grad]
    n_total_params = sum(p.numel() for p in trainable_params) + torch.numel(model.a)

    # embedding = model.a
    # optimizer = torch.optim.Adam([embedding], lr=lr_embedding)
    #
    # _, *parameters = trainable_params
    # for parameter in parameters:
    #     optimizer.add_param_group({'params': parameter, 'lr': lr})

    optimizer = torch.optim.Adam([model.a], lr=lr_embedding)
    for name, parameter in model.named_parameters():
        if (parameter.requires_grad) & (name!='a'):
            optimizer.add_param_group({'params': parameter, 'lr': lr})

    return optimizer, n_total_params

def set_trainable_division_parameters(model, lr):
    trainable_params = [param for _, param in model.named_parameters() if param.requires_grad]
    n_total_params = sum(p.numel() for p in trainable_params) + torch.numel(model.t)

    embedding = model.t
    optimizer = torch.optim.Adam([embedding], lr=lr)

    _, *parameters = trainable_params
    for parameter in parameters:
        optimizer.add_param_group({'params': parameter, 'lr': lr})

    return optimizer, n_total_params

def get_index_particles(x, n_particle_types, dimension):
    index_particles = []
    for n in range(n_particle_types):
        if dimension == 2:
            index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        elif dimension == 3:
            index = np.argwhere(x[:, 7].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())
    return index_particles

def get_type_list(x, dimension):
    type_list = x[:, 1 + 2 * dimension:2 + 2 * dimension].clone().detach()
    return type_list










