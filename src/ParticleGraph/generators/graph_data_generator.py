import matplotlib.pyplot as plt
import numpy as np
import torch
from ParticleGraph.generators.utils import *
from ParticleGraph.models.utils import *
from ParticleGraph.utils import set_size
from scipy import stats
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
from fa2_modified import ForceAtlas2
import torch_geometric.data as data
from torch_geometric.loader import DataLoader

def data_generate(config, visualize=True, run_vizualized=0, style='color', erase=False, step=5, alpha=0.2, ratio=1,
                  scenario='none', device=None, bSave=True):

    has_particle_field = ('PDE_ParticleField' in config.graph_model.particle_model_name)
    has_signal = ('PDE_N' in config.graph_model.signal_model_name)
    has_mesh = (config.graph_model.mesh_model_name != '')
    dataset_name = config.dataset
    print('')
    print(f'dataset_name: {dataset_name}')

    if has_particle_field:
        data_generate_particle_field(config, visualize=visualize, run_vizualized=run_vizualized, style=style, erase=False, step=step,
                                     alpha=0.2, ratio=ratio,
                                     scenario='none', device=None, bSave=True)
    elif has_mesh:
        data_generate_mesh(config, visualize=visualize, run_vizualized=run_vizualized, style=style, erase=erase, step=step,
                                        alpha=0.2, ratio=ratio,
                                        scenario=scenario, device=device, bSave=bSave)
    elif has_signal:
        data_generate_synaptic(config, visualize=visualize, run_vizualized=run_vizualized, style=style, erase=erase, step=step,
                                        alpha=0.2, ratio=ratio,
                                        scenario=scenario, device=device, bSave=bSave)
    else:
        data_generate_particle(config, visualize=visualize, run_vizualized=run_vizualized, style=style, erase=erase, step=step,
                                        alpha=0.2, ratio=ratio,
                                        scenario=scenario, device=device, bSave=bSave)


def data_generate_particle(config, visualize=True, run_vizualized=0, style='color', erase=False, step=5, alpha=0.2, ratio=1, scenario='none', device=None, bSave=True):
    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    print(f'Generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    has_adjacency_matrix = (simulation_config.connectivity_file != '')
    delta_t = simulation_config.delta_t
    n_frames = simulation_config.n_frames
    has_particle_dropout = training_config.particle_dropout > 0
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset

    if config.data_folder_name != 'none':
        print(f'Generating from data ...')
        generate_from_data(config=config, device=device, visualize=visualize, step=step)
        return

    folder = f'./graphs_data/graphs_{dataset_name}/'
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (f[-3:] != 'Fig') & (f[-14:] != 'generated_data') & (f != 'p.pt') & (f != 'cycle_length.pt') & (f != 'model_config.json') & (f != 'generation_code.py'):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/graphs_{dataset_name}/Fig/', exist_ok=True)
    files = glob.glob(f'./graphs_data/graphs_{dataset_name}/Fig/*')
    for f in files:
        os.remove(f)

    # create GNN
    model, bc_pos, bc_dpos = choose_model(config=config, device=device)

    particle_dropout_mask = np.arange(n_particles)
    if has_particle_dropout:
        draw = np.random.permutation(np.arange(n_particles))
        cut = int(n_particles * (1 - training_config.particle_dropout))
        particle_dropout_mask = draw[0:cut]
        inv_particle_dropout_mask = draw[cut:]
        x_removed_list = []

    if simulation_config.angular_Bernouilli != [-1]:
        b = simulation_config.angular_Bernouilli
        generative_m = np.array([stats.norm(b[0], b[2]), stats.norm(b[1], b[2])])

    for run in range(config.training.n_runs):

        n_particles = simulation_config.n_particles

        x_list = []
        y_list = []

        # initialize particle and graph states
        X1, V1, T1, H1, A1, N1 = init_particles(config=config, scenario=scenario, ratio=ratio, device=device)
        time.sleep(0.5)
        for it in trange(simulation_config.start_frame, n_frames + 1):

            x = torch.concatenate(
                (N1.clone().detach(), X1.clone().detach(), V1.clone().detach(), T1.clone().detach(),
                 H1.clone().detach(), A1.clone().detach()), 1)

            index_particles = get_index_particles(x, n_particle_types, dimension)  # can be different from frame to frame

            # compute connectivity rule
            distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
            adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index, field=[])

            # model prediction
            with torch.no_grad():
                y = model(dataset)

            if simulation_config.angular_sigma > 0:
                phi = torch.randn(n_particles, device=device) * simulation_config.angular_sigma / 360 * np.pi * 2
                cos_phi = torch.cos(phi)
                sin_phi = torch.sin(phi)
                new_vx = cos_phi * y[:, 0] - sin_phi * y[:, 1]
                new_vy = sin_phi * y[:, 0] + cos_phi * y[:, 1]
                y = torch.cat((new_vx[:, None], new_vy[:, None]), 1).clone().detach()
            if simulation_config.angular_Bernouilli != [-1]:
                z_i = stats.bernoulli(b[3]).rvs(n_particles)
                phi = np.array([g.rvs() for g in generative_m[z_i]]) / 360 * np.pi * 2
                phi = torch.tensor(phi, device=device, dtype=torch.float32)
                cos_phi = torch.cos(phi)
                sin_phi = torch.sin(phi)
                new_vx = cos_phi * y[:, 0] - sin_phi * y[:, 1]
                new_vy = sin_phi * y[:, 0] + cos_phi * y[:, 1]
                y = torch.cat((new_vx[:, None], new_vy[:, None]), 1).clone().detach()


            # append list
            if (it >= 0) & bSave:
                if has_particle_dropout:
                    x_ = x[particle_dropout_mask].clone().detach()
                    x_[:, 0] = torch.arange(len(x_), device=device)
                    x_list.append(x_)
                    x_ = x[inv_particle_dropout_mask].clone().detach()
                    x_[:, 0] = torch.arange(len(x_), device=device)
                    x_removed_list.append(x[inv_particle_dropout_mask].clone().detach())
                    y_list.append(y[particle_dropout_mask].clone().detach())
                else:
                    x_list.append(x.clone().detach())
                    y_list.append(y.clone().detach())

            # Particle update

            if model_config.prediction == '2nd_derivative':
                V1 += y * delta_t
            else:
                V1 = y
            X1 = bc_pos(X1 + V1 * delta_t)
            A1 = A1 + 1

            # output plots
            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):

                if 'latex' in style:
                    plt.rcParams['text.usetex'] = True
                    rc('font', **{'family': 'serif', 'serif': ['Palatino']})

                if 'bw' in style:

                    fig, ax = fig_init(formatx="%.1f", formaty="%.1f")
                    s_p = 100
                    for n in range(n_particle_types):
                            plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                                        s=s_p, color='k')
                    if training_config.particle_dropout > 0:
                        plt.scatter(x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                    x[inv_particle_dropout_mask, 2].detach().cpu().numpy(), s=25, color='k',
                                    alpha=0.75)
                        plt.plot(x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                 x[inv_particle_dropout_mask, 2].detach().cpu().numpy(), '+', color='w')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    if 'PDE_G' in model_config.particle_model_name:
                        plt.xlim([-2, 2])
                        plt.ylim([-2, 2])
                    if 'latex' in style:
                        plt.xlabel(r'$x$', fontsize=78)
                        plt.ylabel(r'$y$', fontsize=78)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                    elif 'frame' in style:
                        plt.xlabel(r'$x$', fontsize=78)
                        plt.ylabel(r'$y$', fontsize=78)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                    else:
                        plt.xticks([])
                        plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"graphs_data/graphs_{dataset_name}/Fig/Fig_{run}_{it}.jpg", dpi=170.7)
                    plt.close()

                if 'color' in style:

                    if model_config.particle_model_name == 'PDE_O':
                        fig = plt.figure(figsize=(12, 12))
                        plt.scatter(H1[:, 0].detach().cpu().numpy(), H1[:, 1].detach().cpu().numpy(), s=100,
                                    c=np.sin(to_numpy(H1[:, 2])), vmin=-1, vmax=1, cmap='viridis')
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/Fig/Lut_Fig_{run}_{it}.jpg",
                                    dpi=170.7)
                        plt.close()

                        fig = plt.figure(figsize=(12, 12))
                        # plt.scatter(H1[:, 0].detach().cpu().numpy(), H1[:, 1].detach().cpu().numpy(), s=5, c='b')
                        plt.scatter(to_numpy(X1[:, 0]), to_numpy(X1[:, 1]), s=10, c='lawngreen',
                                    alpha=0.75)
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/Fig/Rot_{run}_Fig{it}.jpg",
                                    dpi=170.7)
                        plt.close()

                    elif 'PDE_N' in model_config.signal_model_name:

                        matplotlib.rcParams['savefig.pad_inches'] = 0
                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(1, 1, 1)
                        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                        plt.scatter(to_numpy(X1[:, 1]), to_numpy(X1[:, 0]), s=200, c=to_numpy(H1[:, 0])*3, cmap='viridis')     # vmin=0, vmax=3)
                        plt.colorbar()
                        plt.xlim([-1.2, 1.2])
                        plt.ylim([-1.2, 1.2])
                        # plt.text(0, 1.1, f'frame {it}', ha='left', va='top', transform=ax.transAxes, fontsize=24)
                        # cbar = plt.colorbar(shrink=0.5)
                        # cbar.ax.tick_params(labelsize=32)
                        if 'latex' in style:
                            plt.xlabel(r'$x$', fontsize=78)
                            plt.ylabel(r'$y$', fontsize=78)
                            plt.xticks(fontsize=48.0)
                            plt.yticks(fontsize=48.0)
                        elif 'frame' in style:
                            plt.xlabel('x', fontsize=48)
                            plt.ylabel('y', fontsize=48)
                            plt.xticks(fontsize=48.0)
                            plt.yticks(fontsize=48.0)
                            ax.tick_params(axis='both', which='major', pad=15)
                            plt.text(0, 1.1, f'frame {it}', ha='left', va='top', transform=ax.transAxes, fontsize=48)
                        else:
                            plt.xticks([])
                            plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/Fig/Fig_{run}_{10000 + it}.tif", dpi=70)
                        plt.close()

                    elif (model_config.particle_model_name == 'PDE_A') & (dimension == 3):

                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(111, projection='3d')
                        for n in range(n_particle_types):
                            ax.scatter(to_numpy(x[index_particles[n], 2]), to_numpy(x[index_particles[n], 1]),
                                       to_numpy(x[index_particles[n], 3]), s=50, color=cmap.color(n))
                        ax.set_xlim([0, 1])
                        ax.set_ylim([0, 1])
                        ax.set_zlim([0, 1])
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/Fig/Fig_{run}_{it}.jpg", dpi=170.7)
                        plt.close()

                    else:
                        # matplotlib.use("Qt5Agg")

                        fig, ax = fig_init(formatx="%.1f", formaty="%.1f")
                        s_p = 100
                        for n in range(n_particle_types):
                                plt.scatter(to_numpy(x[index_particles[n], 2]), to_numpy(x[index_particles[n], 1]),
                                            s=s_p, color=cmap.color(n))
                        if training_config.particle_dropout > 0:
                            plt.scatter(x[inv_particle_dropout_mask, 2].detach().cpu().numpy(),
                                        x[inv_particle_dropout_mask, 1].detach().cpu().numpy(), s=25, color='k',
                                        alpha=0.75)
                            plt.plot(x[inv_particle_dropout_mask, 2].detach().cpu().numpy(),
                                     x[inv_particle_dropout_mask, 1].detach().cpu().numpy(), '+', color='w')
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        if 'PDE_G' in model_config.particle_model_name:
                            plt.xlim([-2, 2])
                            plt.ylim([-2, 2])
                        if 'latex' in style:
                            plt.xlabel(r'$x$', fontsize=78)
                            plt.ylabel(r'$y$', fontsize=78)
                            plt.xticks(fontsize=48.0)
                            plt.yticks(fontsize=48.0)
                        if 'frame' in style:
                            plt.xlabel('x', fontsize=48)
                            plt.ylabel('y', fontsize=48)
                            plt.xticks(fontsize=48.0)
                            plt.yticks(fontsize=48.0)
                            ax.tick_params(axis='both', which='major', pad=15)
                            plt.text(0, 1.1, f'frame {it}', ha='left', va='top', transform=ax.transAxes, fontsize=48)
                        if 'no_ticks' in style:
                            plt.xticks([])
                            plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/Fig/Fig_{run}_{it}.tif", dpi=80) # 170.7)
                        # plt.savefig(f"graphs_data/graphs_{dataset_name}/Fig/Fig_{run}_{10000+it}.tif", dpi=42.675)
                        plt.close()

        if bSave:
            torch.save(x_list, f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt')
            if has_particle_dropout:
                torch.save(x_removed_list, f'graphs_data/graphs_{dataset_name}/x_removed_list_{run}.pt')
                np.save(f'graphs_data/graphs_{dataset_name}/particle_dropout_mask.npy', particle_dropout_mask)
                np.save(f'graphs_data/graphs_{dataset_name}/inv_particle_dropout_mask.npy', inv_particle_dropout_mask)
            torch.save(y_list, f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt')
            torch.save(model.p, f'graphs_data/graphs_{dataset_name}/model_p.pt')

    # for handler in logger.handlers[:]:
    #     handler.close()
    #     logger.removeHandler(handler)



def data_generate_synaptic(config, visualize=True, run_vizualized=0, style='color', erase=False, step=5, alpha=0.2, ratio=1, scenario='none', device=None, bSave=True):
    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    print(f'Generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    delta_t = simulation_config.delta_t
    n_frames = simulation_config.n_frames
    has_particle_dropout = training_config.particle_dropout > 0
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset
    is_N2 = 'signal_N2' in dataset_name

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(training_config.seed)

    if config.data_folder_name != 'none':
        print(f'Generating from data ...')
        generate_from_data(config=config, device=device, visualize=visualize, step=step)
        return

    folder = f'./graphs_data/graphs_{dataset_name}/'
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (f[-3:] != 'Fig') & (f[-14:] != 'generated_data') & (f != 'p.pt') & (f != 'cycle_length.pt') & (f != 'model_config.json') & (f != 'generation_code.py'):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/graphs_{dataset_name}/Fig/', exist_ok=True)
    files = glob.glob(f'./graphs_data/graphs_{dataset_name}/Fig/*')
    for f in files:
        os.remove(f)

    particle_dropout_mask = np.arange(n_particles)
    if has_particle_dropout:
        draw = np.random.permutation(np.arange(n_particles))
        cut = int(n_particles * (1 - training_config.particle_dropout))
        particle_dropout_mask = draw[0:cut]
        inv_particle_dropout_mask = draw[cut:]
        x_removed_list = []

    if 'mat' in simulation_config.connectivity_file:
        mat = scipy.io.loadmat(simulation_config.connectivity_file)
        adjacency = torch.tensor(mat['A'], device=device)

        adj_t = adjacency > 0
        edge_index = adj_t.nonzero().t().contiguous()
        edge_attr_adjacency = adjacency[adj_t]

    elif is_N2:
        adjacency = constructRandomMatrices(n_neurons=n_particles, density=1.0, connectivity_mask=f"./graphs_data/{config.simulation.connectivity_mask}" ,device=device)
        adjacency_ = adjacency.t().clone().detach()
        adj_t = torch.abs(adjacency_) > 0
        edge_index = adj_t.nonzero().t().contiguous()
        edge_attr_adjacency = adjacency_[adj_t]

        n_particles = adjacency.shape[0]
        config.simulation.n_particles = n_particles

        # Initial conditions
        X_ = torch.zeros((n_particles, n_frames), device=device)
        Xinit = torch.rand(n_particles, )  # Initial conditions
        X_[:, 0] = Xinit

        torch.save(adjacency, f'./graphs_data/graphs_{dataset_name}/adjacency_asym.pt')

        plt.figure(figsize=(8, 8))
        ax = sns.heatmap(to_numpy(adjacency), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046}, vmin=-0.01, vmax=0.01)
        plt.title('True connectivity matrix', fontsize=12);
        plt.xticks([0, n_particles - 1], [1, n_particles], fontsize=8)
        plt.yticks([0, n_particles - 1], [1, n_particles], fontsize=8)
        plt.tight_layout()
        torch.save(adjacency, f'./graphs_data/graphs_{dataset_name}/adjacency.tif')
        plt.close()

    else:
        mat = scipy.io.loadmat('./graphs_data/Brain.mat')
        first_adjacency = torch.tensor(mat['A'], device=device)
        fig = plt.figure(figsize=(12, 12))
        plt.imshow(to_numpy(first_adjacency>0.01), cmap='viridis')
        plt.colorbar()
        plt.close()

        adjacency = config.simulation.connectivity_init[1] * torch.randn((n_particles, n_particles), dtype=torch.float32, device=device)
        torch.save(adjacency, f'./graphs_data/graphs_{dataset_name}/adjacency_asym.pt')

        i, j = torch.triu_indices(n_particles, n_particles, requires_grad=False, device=device)
        adjacency[i,j] = adjacency[j,i]
        adjacency[i, i] = 0

        # i, j = torch.triu_indices(n_particles, n_particles, requires_grad=False, device=device)
        # bl = adjacency[j,i]
        # pos = torch.argwhere(bl>0)
        # val = bl[pos]
        # indexes = torch.randperm(val.shape[0])
        # bl[pos] = val[indexes]
        # adjacency[j,i] = bl

        adj_t = torch.abs(adjacency) > 0
        edge_index = adj_t.nonzero().t().contiguous()
        edge_attr_adjacency = adjacency[adj_t]
        edge_attr_adjacency_ = to_numpy(edge_attr_adjacency)

        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 2, 1)
        plt.imshow(to_numpy(adjacency), cmap='viridis')
        plt.text (0,-10,f'weight {np.round(np.mean(edge_attr_adjacency_),6)}+/-{np.round(np.std(edge_attr_adjacency_),6)}')

        ax = fig.add_subplot(1, 2, 2)
        plt.imshow(to_numpy(adjacency)>0.01, cmap='viridis')
        plt.text (0,-10,f'{edge_index.shape[1]} edges')
        plt.tight_layout()
        plt.savefig(f"graphs_data/graphs_{dataset_name}/adjacency.tif", dpi=70)
        plt.close()

    # create GNN
    if is_N2:
        match config.simulation.excitation:
            case 'none':
                excitation = torch.ones((n_particles, n_frames+1), device=device) * 0
            case 'constant':
                excitation = torch.ones((n_particles, n_frames+1), device=device) * 0.5
            case 'sine':
                excitation = torch.sin(torch.arange(n_frames+1, device=device)/100) * 2
                excitation = excitation.repeat(n_particles, 1)
        match config.simulation.phi:
            case 'tanh':
                model, bc_pos, bc_dpos = choose_model(config=config, W=adjacency, phi=torch.tanh, device=device)
            case _:
                model, bc_pos, bc_dpos = choose_model(config=config, W=adjacency, phi=torch.tanh, device=device)
    else:
        model, bc_pos, bc_dpos = choose_model(config=config, device=device)
    


    for run in range(config.training.n_runs):

        X = torch.zeros((n_particles, n_frames + 1), device=device)

        x_list = []
        y_list = []

        # initialize particle and graph states
        X1, V1, T1, H1, A1, N1 = init_particles(config=config, scenario=scenario, ratio=ratio, device=device)

        if (is_N2) & (run == 0):
            H1[:,0] = X_[:, 0].clone().detach()

        x = torch.concatenate((N1.clone().detach(), X1.clone().detach(), V1.clone().detach(), T1.clone().detach(), H1.clone().detach(), A1.clone().detach()), 1)

        dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index, edge_attr=edge_attr_adjacency)
        G = to_networkx(dataset, remove_self_loops=True, to_undirected=True)
        forceatlas2 = ForceAtlas2(
            # Behavior alternatives
            outboundAttractionDistribution=True,  # Dissuade hubs
            linLogMode=False,  # NOT IMPLEMENTED
            adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
            edgeWeightInfluence=1.0,

            # Performance
            jitterTolerance=1.0,  # Tolerance
            barnesHutOptimize=True,
            barnesHutTheta=1.2,
            multiThreaded=False,  # NOT IMPLEMENTED

            # Tuning
            scalingRatio=2.0,
            strongGravityMode=False,
            gravity=1.0,

            # Log
            verbose=True)
        positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=500)
        positions = np.array(list(positions.values()))
        X1 = torch.tensor(positions, dtype=torch.float32, device=device)
        X1 = X1 - torch.mean(X1, 0)

        time.sleep(0.5)
        for it in trange(simulation_config.start_frame, n_frames + 1):

            # calculate type change
            if simulation_config.state_type == 'sequence':
                sample = torch.rand((len(T1), 1), device=device)
                sample = (sample < (1 / config.simulation.state_params[0])) * torch.randint(0, n_particle_types,(len(T1), 1), device=device)
                T1 = (T1 + sample) % n_particle_types

            x = torch.concatenate(
                (N1.clone().detach(), X1.clone().detach(), V1.clone().detach(), T1.clone().detach(),
                 H1.clone().detach(), A1.clone().detach()), 1)

            X[:, it] = H1[:, 0].clone().detach()

            dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index, edge_attr=edge_attr_adjacency)

            # model prediction
            with torch.no_grad():
                if is_N2:
                    y, s_tanhu, msg = model(dataset, return_all=True, excitation=excitation[:, it])
                else:
                    y, s_tanhu, msg = model(dataset, return_all=True)

            # append list
            if (it >= 0) & bSave:
                if has_particle_dropout:
                    x_ = x[particle_dropout_mask].clone().detach()
                    x_[:, 0] = torch.arange(len(x_), device=device)
                    x_list.append(x_)
                    x_ = x[inv_particle_dropout_mask].clone().detach()
                    x_[:, 0] = torch.arange(len(x_), device=device)
                    x_removed_list.append(x[inv_particle_dropout_mask].clone().detach())
                    y_list.append(y[particle_dropout_mask].clone().detach())
                else:
                    x_list.append(x.clone().detach())
                    y_list.append(y.clone().detach())

            # Particle update
            if it>=0:
                H1[:, 1] = y.squeeze()
                H1[:, 0] = H1[:, 0] + H1[:, 1] * delta_t

            A1 = A1 + 1

            # output plots
            if visualize & (run <= run_vizualized + 5) & (it % step == 0) & (it >= 0):

                if 'color' in style:

                    matplotlib.rcParams['savefig.pad_inches'] = 0
                    fig = plt.figure(figsize=(13, 10))
                    if 'conn' in config.simulation.connectivity_mask:
                        plt.scatter(to_numpy(X1[:, 1]), to_numpy(X1[:, 0]), s=15, c=to_numpy(H1[:, 0]), cmap='viridis', vmin=-5,vmax=5)
                        plt.colorbar()
                        plt.xlim([-4000, 4000])
                        plt.ylim([-4000, 4000])
                    else:
                        plt.scatter(to_numpy(X1[:, 1]), to_numpy(X1[:, 0]), s=20, c=to_numpy(H1[:, 0]), cmap='viridis')
                        plt.colorbar()
                        plt.xlim([-np.std(positions[:, 0]) / 31, np.std(positions[:, 0]) / 3])
                        plt.ylim([-np.std(positions[:, 1]) / 3, np.std(positions[:, 1]) / 3])
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()

                    # ax = fig.add_subplot(2, 2, 2)
                    # plt.scatter(to_numpy(X1[:, 1]), to_numpy(X1[:, 0]), s=60, c=to_numpy(H1[:, 1]), cmap='viridis', vmin=-2.5, vmax=2.5)
                    # plt.colorbar()
                    # plt.xlim([-1.2, 1.2])
                    # plt.ylim([-1.2, 1.2])
                    # plt.xticks([])
                    # plt.yticks([])
                    # plt.title('du')
                    # ax = fig.add_subplot(2, 2, 3)
                    # plt.scatter(to_numpy(X1[:, 1]), to_numpy(X1[:, 0]), s=60, c=to_numpy(s_tanhu), cmap='viridis', vmin=-2.5, vmax=2.5)
                    # plt.colorbar()
                    # plt.xlim([-1.2, 1.2])
                    # plt.ylim([-1.2, 1.2])
                    # plt.xticks([])
                    # plt.yticks([])
                    # plt.title('s.tanh(u)')
                    # ax = fig.add_subplot(2, 2, 4)
                    # plt.scatter(to_numpy(X1[:, 1]), to_numpy(X1[:, 0]), s=60, c=to_numpy(msg), cmap='viridis', vmin=-2.5, vmax=2.5)
                    # plt.colorbar()
                    # plt.xlim([-1.2, 1.2])
                    # plt.ylim([-1.2, 1.2])
                    # plt.xticks([])
                    # plt.yticks([])
                    # plt.title('g.msg')


                    plt.savefig(f"graphs_data/graphs_{dataset_name}/Fig/Fig_{run}_{10000 + it}.tif", dpi=70)
                    plt.close()

        if (is_N2) & (run==0):
            plt.figure(figsize=(10, 3))
            plt.subplot(121)
            ax = sns.heatmap(to_numpy(X), center=0, cbar_kws={'fraction': 0.046})
            ax.invert_yaxis()
            plt.title('Firing rate', fontsize=12)
            plt.ylabel('Units', fontsize=12)
            plt.xlabel('Time', fontsize=12)
            plt.xticks([])
            plt.yticks([0, 999], [1, 1000], fontsize=12)

            plt.subplot(122)
            plt.title('Firing rate samples', fontsize=12)
            for i in range(50):
                plt.plot(to_numpy(X[i, :]), linewidth=1)
            plt.xlabel('Time', fontsize=12)
            plt.ylabel('Normalized activity', fontsize=12)
            plt.xticks([])
            plt.yticks(fontsize=12)
            plt.tight_layout()
            plt.savefig(f'graphs_data/graphs_{dataset_name}/activity.png', dpi=300)
            plt.close()


        if bSave:
            torch.save(x_list, f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt')
            if has_particle_dropout:
                torch.save(x_removed_list, f'graphs_data/graphs_{dataset_name}/x_removed_list_{run}.pt')
                np.save(f'graphs_data/graphs_{dataset_name}/particle_dropout_mask.npy', particle_dropout_mask)
                np.save(f'graphs_data/graphs_{dataset_name}/inv_particle_dropout_mask.npy', inv_particle_dropout_mask)
            torch.save(y_list, f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt')
            torch.save(model.p, f'graphs_data/graphs_{dataset_name}/model_p.pt')

    # for handler in logger.handlers[:]:
    #     handler.close()
    #     logger.removeHandler(handler)


def data_generate_mesh(config, visualize=True, run_vizualized=0, style='color', erase=False, step=5, alpha=0.2, ratio=1,
                  scenario='none', device=None, bSave=True):
    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    print(f'Generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    delta_t = simulation_config.delta_t
    n_frames = simulation_config.n_frames
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset

    folder = f'./graphs_data/graphs_{dataset_name}/'
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (f[-14:] != 'generated_data') & (f != 'p.pt') & (f != 'cycle_length.pt') & (f != 'model_config.json') & (
                    f != 'generation_code.py'):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/graphs_{dataset_name}/Fig/', exist_ok=True)
    files = glob.glob(f'./graphs_data/graphs_{dataset_name}/Fig/*')
    for f in files:
        os.remove(f)

    for run in range(config.training.n_runs):

        X1_mesh, V1_mesh, T1_mesh, H1_mesh, A1_mesh, N1_mesh, mesh_data = init_mesh(config, device=device)

        mesh_model = choose_mesh_model(config=config, X1_mesh=X1_mesh, device=device)

        torch.save(mesh_data, f'graphs_data/graphs_{dataset_name}/mesh_data_{run}.pt')
        mask_mesh = mesh_data['mask'].squeeze()

        time.sleep(0.5)
        x_mesh_list=[]
        y_mesh_list=[]
        for it in trange(simulation_config.start_frame, n_frames + 1):

            x_mesh = torch.concatenate(
                (N1_mesh.clone().detach(), X1_mesh.clone().detach(), V1_mesh.clone().detach(),
                 T1_mesh.clone().detach(), H1_mesh.clone().detach()), 1)
            x_mesh_list.append(x_mesh.clone().detach())

            dataset_mesh = data.Data(x=x_mesh, edge_index=mesh_data['edge_index'],
                                     edge_attr=mesh_data['edge_weight'], device=device)


            match config.graph_model.mesh_model_name:
                case 'DiffMesh':
                    with torch.no_grad():
                        pred = mesh_model(dataset_mesh)
                        H1[mask_mesh, 1:2] = pred[mask_mesh]
                    H1_mesh[mask_mesh, 0:1] += pred[mask_mesh, 0:1] * delta_t
                    new_pred = torch.zeros_like(pred)
                    new_pred[mask_mesh] = pred[mask_mesh]
                    pred = new_pred
                case 'WaveMesh':
                    with torch.no_grad():
                        pred = mesh_model(dataset_mesh)
                    H1_mesh[mask_mesh, 1:2] += pred[mask_mesh, :] * delta_t
                    H1_mesh[mask_mesh, 0:1] += H1_mesh[mask_mesh, 1:2] * delta_t
                    # x_ = to_numpy(x_mesh)
                    # plt.scatter(x_[:, 1], x_[:, 2], c=to_numpy(H1_mesh[:, 0]))
                case 'RD_Gray_Scott_Mesh' | 'RD_FitzHugh_Nagumo_Mesh' | 'RD_RPS_Mesh' | 'RD_RPS_Mesh_bis':
                    with torch.no_grad():
                        pred = mesh_model(dataset_mesh)
                        H1_mesh[mesh_data['mask'].squeeze(), :] += pred[mesh_data['mask'].squeeze(), :] * delta_t
                        H1_mesh[mask_mesh.squeeze(), 6:9] = torch.clamp(H1_mesh[mask_mesh.squeeze(), 6:9], 0, 1)
                        H1 = H1_mesh.clone().detach()
                case 'PDE_O_Mesh':
                    pred = []

            y_mesh_list.append(pred)

            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):

                # plt.style.use('dark_background')
                # matplotlib.use("Qt5Agg")

                if 'latex' in style:
                    plt.rcParams['text.usetex'] = True
                    rc('font', **{'family': 'serif', 'serif': ['Palatino']})

                if 'graph' in style:

                    fig = plt.figure(figsize=(12, 12))
                    match model_config.mesh_model_name:
                        case 'RD_RPS_Mesh':
                            H1_IM = torch.reshape(x_mesh[:, 6:9], (100, 100, 3))
                            plt.imshow(to_numpy(H1_IM), vmin=0, vmax=1)
                        case 'Wave_Mesh' | 'DiffMesh':
                            pts = x_mesh[:, 1:3].detach().cpu().numpy()
                            tri = Delaunay(pts)
                            colors = torch.sum(x_mesh[tri.simplices, 6], dim=1) / 3.0
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=-2500,
                                          vmax=2500)
                            plt.xlim([0, 1])
                            plt.ylim([0, 1])
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"graphs_data/graphs_{dataset_name}/Fig/Fig_g_color_{it}.tif", dpi=300)
                    plt.close()

                if 'color' in style:

                    # matplotlib.use("Qt5Agg")

                    matplotlib.rcParams['savefig.pad_inches'] = 0
                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_subplot(1, 1, 1)
                    ax.tick_params(axis='both', which='major', pad=15)
                    ax.xaxis.get_major_formatter()._usetex = False
                    ax.yaxis.get_major_formatter()._usetex = False
                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

                    pts = x_mesh[:, 1:3].detach().cpu().numpy()
                    tri = Delaunay(pts)
                    colors = torch.sum(x_mesh[tri.simplices, 6], dim=1) / 3.0
                    match model_config.mesh_model_name:
                        case 'DiffMesh':
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1000)
                        case 'WaveMesh':
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), vmin=-1000, vmax=1000)
                            fmt = lambda x, pos: '{:.1f}'.format((x)/100, pos)
                            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                            plt.xlim([0, 1])
                            plt.ylim([0, 1])
                        case 'RD_Gray_Scott_Mesh':
                            fig = plt.figure(figsize=(12, 6))
                            ax = fig.add_subplot(1, 2, 1)
                            colors = torch.sum(x[tri.simplices, 6], dim=1) / 3.0
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), vmin=0, vmax=1)
                            plt.xticks([])
                            plt.yticks([])
                            plt.axis('off')
                        case 'RD_RPS_Mesh' | 'RD_RPS_Mesh_bis':
                            H1_IM = torch.reshape(H1, (100, 100, 3))
                            plt.imshow(H1_IM.detach().cpu().numpy(), vmin=0, vmax=1)
                            fmt = lambda x, pos: '{:.1f}'.format((x)/100, pos)
                            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                            # plt.xticks([])
                            # plt.yticks([])
                            # plt.axis('off')`
                    if 'latex' in style:
                        plt.xlabel(r'$x$', fontsize=78)
                        plt.ylabel(r'$y$', fontsize=78)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                    elif 'frame' in style:
                        plt.xlabel('x', fontsize=48)
                        plt.ylabel('y', fontsize=48)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                        ax.tick_params(axis='both', which='major', pad=15)
                        plt.text(0, 1.1, f'frame {it}', ha='left', va='top', transform=ax.transAxes, fontsize=48)
                    else:
                        plt.xticks([])
                        plt.yticks([])

                    plt.tight_layout()
                    plt.savefig(f"graphs_data/graphs_{dataset_name}/Fig/Fig_{run}_{it}.tif", dpi=170.7)
                    # plt.savefig(f"graphs_data/graphs_{dataset_name}/Fig/Fig_{run}_{10000+it}.tif", dpi=42.675)
                    plt.close()

        if bSave:
            torch.save(x_mesh_list, f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt')
            torch.save(y_mesh_list, f'graphs_data/graphs_{dataset_name}/y_mesh_list_{run}.pt')



def data_generate_particle_field(config, visualize=True, run_vizualized=0, style='color', erase=False, step=5, alpha=0.2, ratio=1,
                  scenario='none', device=None, bSave=True):
    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    print(f'Generating data ... {config} {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    n_nodes = simulation_config.n_nodes
    n_nodes_per_axis = int(np.sqrt(n_nodes))
    delta_t = simulation_config.delta_t
    has_adjacency_matrix = (simulation_config.connectivity_file != '')
    n_frames = simulation_config.n_frames
    has_particle_dropout = training_config.particle_dropout > 0
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset

    folder = f'./graphs_data/graphs_{dataset_name}/'
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (f[-14:] != 'generated_data') & (f != 'p.pt') & (f != 'cycle_length.pt') & (f != 'model_config.json') & (
                    f != 'generation_code.py'):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/graphs_{dataset_name}/Fig/', exist_ok=True)
    files = glob.glob(f'./graphs_data/graphs_{dataset_name}/Fig/*')
    for f in files:
        os.remove(f)
    copyfile(os.path.realpath(__file__), os.path.join(folder, 'generation_code.py'))

    model_p_p, bc_pos, bc_dpos = choose_model(config=config, device=device)
    model_f_p = model_p_p

    # model_f_f = choose_mesh_model(config, device=device)

    index_particles = []
    for n in range(n_particle_types):
        index_particles.append(
            np.arange((n_particles // n_particle_types) * n, (n_particles // n_particle_types) * (n + 1)))
    if has_particle_dropout:
        draw = np.random.permutation(np.arange(n_particles))
        cut = int(n_particles * (1 - training_config.particle_dropout))
        particle_dropout_mask = draw[0:cut]
        inv_particle_dropout_mask = draw[cut:]
        x_removed_list = []
    else:
        particle_dropout_mask = np.arange(n_particles)
    if has_adjacency_matrix:
        mat = scipy.io.loadmat(simulation_config.connectivity_file)
        adjacency = torch.tensor(mat['A'], device=device)
        adj_t = adjacency > 0
        edge_index = adj_t.nonzero().t().contiguous()
        edge_attr_adjacency = adjacency[adj_t]

    for run in range(config.training.n_runs):

        n_particles = simulation_config.n_particles

        x_list = []
        y_list = []
        x_mesh_list = []
        y_mesh_list = []
        edge_p_p_list = []
        edge_f_p_list = []

        # initialize particle and mesh states
        X1, V1, T1, H1, A1, N1 = init_particles(config=config, scenario=scenario, ratio=ratio, device=device)
        # X1_mesh, V1_mesh, T1_mesh, H1_mesh, A1_mesh, N1_mesh, mesh_data = init_mesh(config, model_mesh=model_f_f, device=device)
        X1_mesh, V1_mesh, T1_mesh, H1_mesh, A1_mesh, N1_mesh, mesh_data = init_mesh(config, device=device)

        # matplotlib.use("Qt5Agg")
        # fig = plt.figure(figsize=(12, 12))
        # im = torch.reshape(H1_mesh[:,0:1],(100,100))
        # plt.imshow(to_numpy(im))
        # plt.colorbar()

        torch.save(mesh_data, f'graphs_data/graphs_{dataset_name}/mesh_data_{run}.pt')
        mask_mesh = mesh_data['mask'].squeeze()

        time.sleep(0.5)
        for it in trange(simulation_config.start_frame, n_frames + 1):

            if ('siren' in model_config.field_type) & (it >= 0):
                im = imread(f"graphs_data/{simulation_config.node_value_map}") # / 255 * 5000
                im = im[it].squeeze()
                im = np.rot90(im,3)
                im = np.reshape(im, (n_nodes_per_axis * n_nodes_per_axis))
                H1_mesh[:, 0:1] = torch.tensor(im[:,None], dtype=torch.float32, device=device)

            x = torch.concatenate((N1.clone().detach(), X1.clone().detach(), V1.clone().detach(), T1.clone().detach(),
                                   H1.clone().detach(), A1.clone().detach()), 1)

            index_particles = get_index_particles(x, n_particle_types, dimension)

            x_mesh = torch.concatenate(
                (N1_mesh.clone().detach(), X1_mesh.clone().detach(), V1_mesh.clone().detach(),
                 T1_mesh.clone().detach(), H1_mesh.clone().detach(), A1_mesh.clone().detach()), 1)

            x_particle_field = torch.concatenate((x_mesh, x), dim=0)

            # compute connectivity rules
            distance = torch.sum(bc_dpos(x[:, None, 1:dimension+1] - x[None, :, 1:dimension+1]) ** 2, dim=2)
            adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            dataset_p_p = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)
            if not(has_particle_dropout):
                edge_p_p_list.append(edge_index)

            distance = torch.sum(bc_dpos(x_particle_field[:, None, 1:dimension+1] - x_particle_field[None, :, 1:dimension+1]) ** 2, dim=2)
            adj_t = ((distance < (max_radius/2) ** 2) & (distance > min_radius ** 2)).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            pos = torch.argwhere((edge_index[1,:]>=n_nodes) & (edge_index[0,:]<n_nodes))
            pos = to_numpy(pos[:,0])
            edge_index = edge_index[:,pos]
            dataset_f_p = data.Data(x=x_particle_field, pos=x_particle_field[:, 1:3], edge_index=edge_index)
            if not (has_particle_dropout):
                edge_f_p_list.append(edge_index)

            # model prediction
            with torch.no_grad():
                y0 = model_p_p(dataset_p_p,has_field=False)
                y1 = model_f_p(dataset_f_p,has_field=True)[n_nodes:]
                y = y0 + y1

            # append list
            if (it >= 0) & bSave:
                if has_particle_dropout:

                    x_ = x[inv_particle_dropout_mask].clone().detach()
                    x_[:, 0] = torch.arange(len(x_), device=device)
                    x_removed_list.append(x[inv_particle_dropout_mask].clone().detach())
                    x_ = x[particle_dropout_mask].clone().detach()
                    x_[:, 0] = torch.arange(len(x_), device=device)
                    x_list.append(x_)
                    y_list.append(y[particle_dropout_mask].clone().detach())

                    distance = torch.sum(bc_dpos(x_[:, None, 1:dimension + 1] - x_[None, :, 1:dimension + 1]) ** 2, dim=2)
                    adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
                    edge_index = adj_t.nonzero().t().contiguous()
                    edge_p_p_list.append(edge_index)

                    x_particle_field = torch.concatenate((x_mesh, x_), dim=0)

                    distance = torch.sum(bc_dpos(
                        x_particle_field[:, None, 1:dimension + 1] - x_particle_field[None, :, 1:dimension + 1]) ** 2, dim=2)
                    adj_t = ((distance < (max_radius / 2) ** 2) & (distance > min_radius ** 2)).float() * 1
                    edge_index = adj_t.nonzero().t().contiguous()
                    pos = torch.argwhere((edge_index[1, :] >= n_nodes) & (edge_index[0, :] < n_nodes))
                    pos = to_numpy(pos[:, 0])
                    edge_index = edge_index[:, pos]
                    edge_f_p_list.append(edge_index)

                else:
                    x_list.append(x.clone().detach())
                    y_list.append(y.clone().detach())

            # Particle update
            if model_config.prediction == '2nd_derivative':
                V1 += y * delta_t
            else:
                V1 = y
            X1 = bc_pos(X1 + V1 * delta_t)

            A1 = A1 + 1

            # Mesh update
            x_mesh_list.append(x_mesh.clone().detach())
            pred = x_mesh[:,6:7]
            y_mesh_list.append(pred)

            # output plots
            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):

                # plt.style.use('dark_background')
                # matplotlib.use("Qt5Agg")

                if 'latex' in style:
                    plt.rcParams['text.usetex'] = True
                    rc('font', **{'family': 'serif', 'serif': ['Palatino']})

                if 'graph' in style:

                    fig = plt.figure(figsize=(10, 10))

                    if model_config.mesh_model_name == 'RD_RPS_Mesh':
                        H1_IM = torch.reshape(x_mesh[:, 6:9], (100, 100, 3))
                        plt.imshow(to_numpy(H1_IM), vmin=0, vmax=1)
                    elif (model_config.mesh_model_name == 'Wave_Mesh') | (model_config.mesh_model_name =='DiffMesh') :
                        pts = x_mesh[:, 1:3].detach().cpu().numpy()
                        tri = Delaunay(pts)
                        colors = torch.sum(x_mesh[tri.simplices, 6], dim=1) / 3.0
                        if model_config.mesh_model_name == 'WaveMesh' :
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=-2500,
                                          vmax=2500)
                        else:
                            plt.tripcolor(pts[:, 0], pts[:, 1], tri.simplices.copy(),
                                          facecolors=colors.detach().cpu().numpy(), edgecolors='k', vmin=0, vmax=5000)
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                    elif model_config.particle_model_name == 'PDE_G':
                        for n in range(n_particle_types):
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=40, color=cmap.color(n))
                    elif model_config.particle_model_name == 'PDE_E':
                        for n in range(n_particle_types):
                            g = 40
                            if simulation_config.params[n][0] <= 0:
                                plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=g, c=cmap.color(n))
                            else:
                                plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                            x[index_particles[n], 2].detach().cpu().numpy(), s=g, c=cmap.color(n))
                    else:
                        for n in range(n_particle_types):
                            plt.scatter(x[index_particles[n], 1].detach().cpu().numpy(),
                                        x[index_particles[n], 2].detach().cpu().numpy(), s=25, color=cmap.color(n),
                                        alpha=0.5)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"graphs_data/graphs_{dataset_name}/Fig/Fig_g_color_{it}.tif", dpi=300)
                    plt.close()

                if 'bw' in style:

                    matplotlib.rcParams['savefig.pad_inches'] = 0
                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_subplot(1, 1, 1)
                    s_p = 50
                    if False:  # config.simulation.non_discrete_level>0:
                        plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), s=s_p, color='k')
                    else:
                        for n in range(n_particle_types):
                            plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                                        s=s_p, color='k')
                    if training_config.particle_dropout > 0:
                        plt.scatter(x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                    x[inv_particle_dropout_mask, 2].detach().cpu().numpy(), s=25, color='k',
                                    alpha=0.75)
                        plt.plot(x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                 x[inv_particle_dropout_mask, 2].detach().cpu().numpy(), '+', color='w')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"graphs_data/graphs_{dataset_name}/Fig/Fig_{run}_{it}.jpg", dpi=170.7)
                    plt.close()

                if 'color' in style:

                    # matplotlib.use("Qt5Agg")
                    matplotlib.rcParams['savefig.pad_inches'] = 0
                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_subplot(1, 1, 1)
                    # ax.xaxis.get_major_formatter()._usetex = False
                    # ax.yaxis.get_major_formatter()._usetex = False
                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax.tick_params(axis='both', which='major', pad=15)
                    # if (has_mesh | (simulation_config.boundary == 'periodic')):
                    #     ax = plt.axes([0, 0, 1, 1], frameon=False)
                    # else:
                    #     ax = plt.axes([-2, -2, 2, 2], frameon=False)
                    # ax.get_xaxis().set_visible(False)
                    # ax.get_yaxis().set_visible(False)
                    # plt.autoscale(tight=True)
                    s_p = 50
                    for n in range(n_particle_types):
                            plt.scatter(to_numpy(x[index_particles[n], 2]), to_numpy(x[index_particles[n], 1]),
                                        s=s_p, color=cmap.color(n))
                    plt.xlim([0,1])
                    plt.ylim([0,1])
                    # plt.xlim([-2,2])
                    # plt.ylim([-2,2])
                    if 'latex' in style:
                        plt.xlabel(r'$x$', fontsize=78)
                        plt.ylabel(r'$y$', fontsize=78)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                    elif 'frame' in style:
                        plt.xlabel('x', fontsize=78)
                        plt.ylabel('y', fontsize=78)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                    else:
                        plt.xticks([])
                        plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"graphs_data/graphs_{dataset_name}/Fig/Fig_{run}_{it}.jpg", dpi=170.7)
                    plt.close()

                    matplotlib.rcParams['savefig.pad_inches'] = 0

                    if model_config.prediction == '2nd_derivative':
                        V0_ = y0  * delta_t
                        V1_ = y1  * delta_t
                    else:
                        V0_ = y0
                        V1_ = y1
                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_subplot(1, 1, 1)
                    type_list = to_numpy(get_type_list(x, dimension))
                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax.tick_params(axis='both', which='major', pad=15)
                    plt.scatter(to_numpy(x_mesh[0:n_nodes, 2]), to_numpy(x_mesh[0:n_nodes, 1]), c=to_numpy(x_mesh[0:n_nodes, 6]),cmap='grey',s=5)
                    plt.xlim([0,1])
                    plt.ylim([0,1])
                    for n in range(n_particles):
                        plt.arrow(x=to_numpy(x[n, 2]), y=to_numpy(x[n, 1]), dx=to_numpy(V1_[n,1])*4.25, dy=to_numpy(V1_[n,0])*4.25, color=cmap.color(type_list[n].astype(int)), head_width=0.004, length_includes_head=True)
                    # plt.xlim([-2,2])
                    # plt.ylim([-2,2])
                    if 'latex' in style:
                        plt.xlabel(r'$x$', fontsize=78)
                        plt.ylabel(r'$y$', fontsize=78)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                    elif 'frame' in style:
                        plt.xlabel('x', fontsize=78)
                        plt.ylabel('y', fontsize=78)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                    else:
                        plt.xticks([])
                        plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(f"graphs_data/graphs_{dataset_name}/Fig/Arrow_{run}_{it}.jpg", dpi=170.7)
                    plt.close()

        if bSave:
            torch.save(x_list, f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt')
            if has_particle_dropout:
                torch.save(x_removed_list, f'graphs_data/graphs_{dataset_name}/x_removed_list_{run}.pt')
                np.save(f'graphs_data/graphs_{dataset_name}/particle_dropout_mask.npy', particle_dropout_mask)
                np.save(f'graphs_data/graphs_{dataset_name}/inv_particle_dropout_mask.npy', inv_particle_dropout_mask)
            torch.save(y_list, f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt')
            torch.save(x_mesh_list, f'graphs_data/graphs_{dataset_name}/x_mesh_list_{run}.pt')
            torch.save(y_mesh_list, f'graphs_data/graphs_{dataset_name}/y_mesh_list_{run}.pt')
            torch.save(edge_p_p_list, f'graphs_data/graphs_{dataset_name}/edge_p_p_list{run}.pt')
            torch.save(edge_f_p_list, f'graphs_data/graphs_{dataset_name}/edge_f_p_list{run}.pt')
            torch.save(model_p_p.p, f'graphs_data/graphs_{dataset_name}/model_p.pt')


