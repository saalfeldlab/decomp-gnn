import matplotlib.pyplot as plt
import numpy as np
import torch
from ParticleGraph.generators.utils import *
from ParticleGraph.models.utils import *
from ParticleGraph.utils import set_size
from ParticleGraph.generators.cell_utils import *
from scipy import stats
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
from fa2_modified import ForceAtlas2

def data_generate(config, visualize=True, run_vizualized=0, style='color', erase=False, step=5, alpha=0.2, ratio=1,
                  scenario='none', device=None, bSave=True):

    has_particle_field = ('PDE_ParticleField' in config.graph_model.particle_model_name)
    has_signal = ('PDE_N' in config.graph_model.signal_model_name)
    has_mesh = (config.graph_model.mesh_model_name != '')
    has_cell_divsion = config.simulation.has_cell_division
    has_fluo = config.simulation.has_fluo
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
    elif has_fluo:
        data_generate_cell_from_fluo(config, visualize=visualize, run_vizualized=run_vizualized, style=style, erase=erase,
                           step=step,
                           alpha=0.2, ratio=ratio,
                           scenario=scenario, device=device, bSave=bSave)
    elif has_cell_divsion:
            data_generate_cell(config, visualize=visualize, run_vizualized=run_vizualized, style=style, erase=erase, step=step,
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

            # calculate type change
            if simulation_config.state_type == 'sequence':
                sample = torch.rand((len(T1), 1), device=device)
                sample = (sample < (1 / config.simulation.state_params[0])) * torch.randint(0, n_particle_types,(len(T1), 1), device=device)
                T1 = (T1 + sample) % n_particle_types

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



def data_generate_cell(config, visualize=True, run_vizualized=0, style='color', erase=False, step=5, alpha=0.2,
                       ratio=1, scenario='none', device=None, bSave=True):
    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(42)

    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    print(f'Generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles_max = simulation_config.n_particles_max
    delta_t = simulation_config.delta_t
    n_frames = simulation_config.n_frames
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset
    marker_size = config.plotting.marker_size
    has_inert_model = simulation_config.cell_inert_model_coeff > 0
    has_cell_death = simulation_config.has_cell_death
    has_cell_division = True

    max_radius_list = []
    edges_len_list = []
    folder = f'./graphs_data/graphs_{dataset_name}/'
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/graphs_{dataset_name}/Fig/', exist_ok=True)
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (f[-3:] != 'Fig') & (f[-2:] != 'GT') & (f != 'p.pt') & (f != 'cycle_length.pt') & (f != 'model_config.json') & (f != 'generation_code.py'):
                os.remove(f)
        files = glob.glob(f'./graphs_data/graphs_{dataset_name}/Fig/*')
        for f in files:
            os.remove(f)


    logging.basicConfig(filename=f'./graphs_data/graphs_{dataset_name}/generator.log', format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(config)

    kill_cell_leaving = False

    for run in range(config.training.n_runs-1):

        torch.cuda.empty_cache()

        model, bc_pos, bc_dpos = choose_model(config=config, device=device)

        n_particles = simulation_config.n_particles

        x_list = []
        y_list = []
        area_list = []
        d_pos = []
        x_len_list = []
        edge_p_p_list = []
        vertices_pos_list = []
        vertices_per_cell_list = []
        current_loss =[]

        '''
        INITIALIZE PER CELL TYPE VALUES1000
        cycle_length
        final_cell_mass
        cell_death_rate
        mc_slope: scalar used in the function set_mass_coeff

        INITIALIZE PER CELL VALUES
        0 N1 cell index dim=1
        1,2 X1 positions dim=2
        3,4 V1 velocities dim=2
        5 T1 cell type dim=1
        6,7 H1 cell status dim=2  H1[:,0] = cell alive flag, alive : 0 , death : 0 , H1[:,1] = cell division flag, dividing : 1
        8 A1 cell age dim=1
        9 S1 cell stage dim=1  0 = G1 , 1 = S, 2 = G2, 3 = M
        10 M1 cell_mass_distrib mass dim=1 (per node)
        11 R1 cell growth rate dim=1
        12 DR1 cell death rate dim=1
        13 MC1 mass coefficient of the cell (relation between velocity and mass)
        14 AR1 area of the cell
        '''

        if run == 0:
            cycle_length, final_cell_mass, cell_death_rate, mc_slope, cell_area = init_cell_range(config, device=device)

        N1, X1, V1, T1, H1, A1, S1, M1, R1, CL1, DR1, MC1, AR1, P1 = init_cells(config, cycle_length, final_cell_mass,
                                                                            cell_death_rate, mc_slope, cell_area, bc_pos, bc_dpos, dimension,
                                                                            device=device)

        coeff = 0
        num_cells = []
        for i in range(n_particle_types):
            pos = torch.argwhere(T1.squeeze() == i).shape[0]
            num_cells.append(pos)
            coeff += num_cells[i] * cell_area[i]
        target_areas_per_type = torch.tensor([cell_area[i] / coeff for i in range(n_particle_types)], device=device)
        target_areas = target_areas_per_type[to_numpy(T1).astype(int)].squeeze().clone().detach()

        T1_list = T1.clone().detach()

        man_track = torch.cat((N1 + 1, torch.zeros((len(N1), 3), device=device)), 1)
        man_track[:, 2] = -1

        logger.info('cell cycle length')
        logger.info(to_numpy(cycle_length))
        logger.info('cell death rate')
        logger.info(to_numpy(cell_death_rate))
        logger.info("cell final mass")
        logger.info(to_numpy(final_cell_mass))
        logger.info('mass coefficient ranges')
        logger.info(mc_slope)
        logger.info('interaction parameters')
        logger.info(to_numpy(model.p))


        index_particles = []
        for n in range(n_particle_types):
            pos = torch.argwhere(T1.squeeze() == n)
            pos = to_numpy(pos[:, 0].squeeze()).astype(int)
            index_particles.append(pos)
        n_particles_alive = len(X1)
        n_particles_dead = 0

        time.sleep(0.5)
        for it in trange(simulation_config.start_frame, n_frames + 1):

            # calculate cell death and cell division

            if has_cell_death:
                # cell death
                sample = torch.rand(len(X1), device=device)
                if kill_cell_leaving:
                    H1[X1[:,0]<0] = 0
                    H1[X1[:,0]>1] = 0
                    H1[X1[:,1]<0] = 0
                    H1[X1[:,1]>1] = 0
                H1[sample.squeeze() < DR1.squeeze() / 5E4, 0] = 0
                # removal if too small
                pos = torch.argwhere((AR1.squeeze()<2E-4) & (A1.squeeze() > 25))
                if len(pos) > 0:
                    H1[pos,0]=0
                    man_track[to_numpy(N1[pos]).astype(int), 2] = it - 1
                n_particles_alive = torch.sum(H1[:, 0])
                n_particles_dead = n_particles - n_particles_alive


            if (it > 0) & (has_cell_division):

                # cell division
                pos = torch.argwhere(
                    (A1.squeeze() >= CL1.squeeze()) & (H1[:, 0].squeeze() == 1) & (S1[:, 0].squeeze() == 3) & (n_particles_alive < n_particles_max)).flatten()
                if (len(pos) > 0):
                    n_add_nodes = len(pos) * 2
                    pos = to_numpy(pos).astype(int)

                    N1_ = n_particles + torch.arange(n_add_nodes, device=device)
                    N1 = torch.cat((N1, N1_[:, None]), dim=0)

                    # man_track = tracklet ID, start time, end time, parent tracklet
                    man_track_ = torch.cat((N1_[:, None] + 1, torch.zeros((n_add_nodes, 3), device=device)),1)  # cell ID
                    man_track_[:, 1] = it  # start time
                    man_track_[:, 2] = -1  # end time
                    man_track_[0:n_add_nodes // 2, 3:4] = N1[pos] + 1  # parent cell
                    man_track_[n_add_nodes // 2:n_add_nodes, 3:4] = N1[pos] + 1  # parent cell
                    man_track = torch.cat((man_track, man_track_), 0)
                    man_track[to_numpy(N1[pos]).astype(int), 2] = it - 1  # end time

                    n_particles = n_particles + n_add_nodes

                    angle = torch.atan(V1[pos, 1] / (V1[pos, 0] + 1E-10))
                    separation = [torch.cos(angle) * 0.005,torch.sin(angle) * 0.005]
                    separation = torch.stack(separation)
                    separation = separation.t()

                    X1 = torch.cat((X1, X1[pos, :] + separation, X1[pos, :] - separation), dim=0)

                    nd = torch.ones(len(pos), device=device) + 0.05 * torch.randn(len(pos), device=device)
                    var = torch.ones(len(pos), device=device) + 0.20 * torch.randn(len(pos), device=device)

                    V1 = torch.cat((V1, V1[pos, :], -V1[pos, :]), dim=0)  # the new cell is moving away from its mother
                    T1 = torch.cat((T1, T1[pos, :], T1[pos, :]), dim=0)

                    T1_list = torch.cat((T1_list, T1[pos, :], T1[pos, :]), dim=0)

                    H1[pos, 0] = 0  # mother cell is removed, considered dead
                    H1[pos, 1] = 1  # cell division flag
                    H1 = torch.concatenate((H1, torch.ones((n_add_nodes, 2), device=device)), 0)
                    H1 [-n_add_nodes:, 1] = 0  # cell division flag = 0 for new daughter cells
                    A1 = torch.cat((A1, torch.ones((n_add_nodes, 1), device=device)), 0)
                    S1 = torch.cat((S1, torch.ones((n_add_nodes, 1), device=device)), 0)
                    M1 = torch.cat((M1, final_cell_mass[to_numpy(T1[pos, 0]), None] / 2,
                                    final_cell_mass[to_numpy(T1[pos, 0]), None] / 2), dim=0)
                    CL1 = torch.cat((CL1, cycle_length[to_numpy(T1[pos, 0]), None] * var[:, None],
                                     cycle_length[to_numpy(T1[pos, 0]), None] * var[:, None]), dim=0)
                    DR1 = torch.cat((DR1, cell_death_rate[to_numpy(T1[pos, 0]), None] * nd[:, None],
                                     cell_death_rate[to_numpy(T1[pos, 0]), None] * nd[:, None]), dim=0)
                    MC1 = torch.cat((MC1, mc_slope[to_numpy(T1[pos, 0]), None], mc_slope[to_numpy(T1[pos, 0]), None]), dim=0)
                    AR1 = torch.cat((AR1, AR1[pos, :], AR1[pos, :]), dim=0)
                    P1 = torch.cat((P1, P1[pos, :], P1[pos, :]), dim=0)
                    R1 = M1 / (2 * CL1)

                    target_areas = torch.cat((target_areas, target_areas[pos], target_areas[pos]), dim=0)

                    n_particles_alive = torch.sum(H1[:, 0])

                    if (n_particles_alive >= simulation_config.n_particles_max):
                        has_cell_division = False
                        has_cell_death = False

            alive = torch.argwhere(H1[:, 0] == 1).squeeze()

            N1 = N1[alive]
            X1 = X1[alive]
            V1 = V1[alive]
            T1 = T1[alive]
            H1 = H1[alive]
            A1 = A1[alive]
            S1 = S1[alive]
            M1 = M1[alive]
            CL1 = CL1[alive]
            R1 = R1[alive]
            DR1 = DR1[alive]
            MC1 = MC1[alive]
            AR1 = AR1[alive]
            P1 = P1[alive]
            target_areas = target_areas[alive]

            index_particles = []
            for n in range(n_particle_types):
                pos = torch.argwhere(T1.squeeze() == n)
                pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                index_particles.append(pos)

            # calculate cell type change
            if simulation_config.state_type == 'sequence':
                sample = torch.rand((len(T1), 1), device=device)
                sample = (sample < (1 / config.simulation.state_params[0])) * torch.randint(0, n_particle_types,
                                                                                            (len(T1), 1), device=device)
                T1 = (T1 + sample) % n_particle_types

            A1 = A1 + delta_t  # update age

            if n_particles_alive < n_particles_max:
                S1 = update_cell_cycle_stage(A1, cycle_length, T1, device)
                M1 += R1 * delta_t

            if (it==simulation_config.start_frame):
                ID1 = torch.arange(len(N1), device=device)[:, None]
            else:
                ID1 = torch.arange(int(ID1[-1]+1), int(ID1[-1]+len(N1)+1),device=device)[:, None]

            x = torch.concatenate((N1.clone().detach(), X1.clone().detach(), V1.clone().detach(), T1.clone().detach(),
                                   H1.clone().detach(), A1.clone().detach(), S1.clone().detach(), M1.clone().detach(),
                                   R1.clone().detach(), DR1.clone().detach(), MC1.clone().detach(),
                                   AR1.clone().detach(), P1.clone().detach(), ID1.clone().detach() ), 1)

            # calculate connectivity
            with torch.no_grad():
                edge_index = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                edge_index = ((edge_index < max_radius ** 2) & (edge_index > min_radius ** 2)).float() * 1
                edge_index = edge_index.nonzero().t().contiguous()
                edge_p_p_list.append(to_numpy(edge_index))
                alive = (H1[:, 0] == 1).float() * 1.
                dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)
                if edge_index.shape[1] > simulation_config.max_edges:
                    max_radius = max_radius / 1.025
                else:
                    max_radius = max_radius * 1.0025
                max_radius = np.clip(max_radius, simulation_config.min_radius, simulation_config.max_radius)
                max_radius_list.append(max_radius)
                edges_len_list.append(edge_index.shape[1])
                x_len_list.append(x.shape[0])

            # model prediction
            with torch.no_grad():
                y = model(dataset, has_field=True)
                y = y * alive[:, None].repeat(1, 2) * simulation_config.cell_active_model_coeff

            first_X1 = X1.clone().detach()

            if has_inert_model:

                X1_ = X1.clone().detach()
                X1_.requires_grad = True

                optimizer = torch.optim.Adam([X1_], lr=1E-3)
                optimizer.zero_grad()
                vor, vertices_pos, vertices_per_cell, all_points = get_vertices(points=X1_, device=device)
                cc, tri = get_Delaunay(all_points,device)
                distance = torch.sum((vertices_pos[:, None, :].clone().detach() - cc[None, :, :]) ** 2, dim=2)
                result = distance.min(dim=1)
                index = result.indices
                cc = cc[index]
                # tri = tri[index]

                voronoi_area = get_voronoi_areas(cc, vertices_per_cell, device)
                perimeter = get_voronoi_perimeters(cc, vertices_per_cell, device)
                AR1 = voronoi_area[:, None].clone().detach()
                P1 = perimeter[:,None].clone().detach()

                loss = simulation_config.coeff_area * (target_areas - voronoi_area).norm(2)
                if simulation_config.coeff_perimeter>0:
                    loss += simulation_config.coeff_perimeter * torch.sum(perimeter**2)

                loss.backward()
                optimizer.step()

                # print(f'loss {loss.item()}')
                # fig = plt.figure(figsize=(12, 12))
                # ax = fig.add_subplot(1, 1, 1)
                # vor, vertices_pos, vertices_per_cell, all_points = get_vertices(points=X1, device=device)
                # voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=1, line_alpha=0.5,
                #                 point_size=0)
                # plt.scatter(to_numpy(cc[:, 0]), to_numpy(cc[:, 1]), s=1, color='r')
                #
                # fig = plt.figure(figsize=(12, 12))
                # ax = fig.add_subplot(1, 1, 1)
                # vor, vertices_pos, vertices_per_cell, all_points = get_vertices(points=X1, device=device)
                # voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=1, line_alpha=0.5,
                #                 point_size=0)
                # vor, vertices_pos, vertices_per_cell, all_points = get_vertices(points=X1_, device=device)
                # voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='red', line_width=1, line_alpha=0.5,
                #                 point_size=0)

                current_loss.append(loss.item())

                X1 = bc_pos(X1_.clone().detach())

            if model_config.prediction == '2nd_derivative':
                y_voronoi = (bc_dpos(X1 - first_X1) / delta_t - V1) / delta_t * simulation_config.cell_inert_model_coeff
            else:
                y_voronoi = bc_dpos(X1 - first_X1) / delta_t * simulation_config.cell_inert_model_coeff

            if (it) % 100 == 0:
                t, r, a = get_gpu_memory_map(device)
                logger.info(f"GPU memory: total {t} reserved {r} allocated {a}")
                logger.info(f'{x.shape[0]} particles,  {edge_index.shape[1]} edges, max_radius: {np.round(max_radius, 3)}')
                memory_usage_x_list=0
                memory_usage_edge_list=0
                for n in range(len(x_list)):
                    memory_usage_x_list += x_list[n].element_size() * x_list[n].nelement()
                    memory_usage_edge_list += edge_p_p_list[n].nbytes
                logger.info(f'memory usage x_list: {memory_usage_x_list/1E6} MB, memory usage edge_list: {memory_usage_edge_list/1E6} MB')
                print(f"GPU memory: total {t} reserved {r} allocated {a}")
                print(f'{x.shape[0]} particles,  {edge_index.shape[1]} edges, max_radius: {np.round(max_radius, 3)}')
                print(f'memory usage x_list: {memory_usage_x_list/1E6} MB, memory usage edge_list: {memory_usage_edge_list/1E6} MB')

            # append list
            if (it >= 0):
                x_list.append(x)
                y_list.append(y + y_voronoi)

            # get mass_coeff
            # mass_coeff = set_mass_coeff(mc_slope_distrib, cell_mass[to_numpy(T1[:, 0])], M1, device)

            # cell update
            if model_config.prediction == '2nd_derivative':
                V1 += (y + y_voronoi) * delta_t
            else:
                V1 = y + y_voronoi

            if kill_cell_leaving:
                X1 = first_X1 + V1 * delta_t
            else:
                X1 = bc_pos(X1 + V1 * delta_t)

            if has_inert_model:
                vor, vertices_pos, vertices_per_cell, all_points = get_vertices(points=X1, device=device)
                vertices_pos_list.append(to_numpy(vertices_pos))

            # output plots
            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):

                # plt.style.use('dark_background')
                # matplotlib.use("Qt5Agg")

                if 'latex' in style:
                    plt.rcParams['text.usetex'] = True
                    rc('font', **{'family': 'serif', 'serif': ['Palatino']})

                if 'bw' in style:

                    matplotlib.rcParams['savefig.pad_inches'] = 0
                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_subplot(1, 1, 1)
                    for n in range(n_particle_types):
                        plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                                    s=marker_size, color='k')
                    if training_config.particle_dropout > 0:
                        plt.scatter(x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                    x[inv_particle_dropout_mask, 2].detach().cpu().numpy(), s=25, color='k',
                                    alpha=0.75)
                        plt.plot(x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                 x[inv_particle_dropout_mask, 2].detach().cpu().numpy(), '+', color='w')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    if 'latex' in style:
                        plt.xlabel(r'$x$', fontsize=78)
                        plt.ylabel(r'$y$', fontsize=78)
                        plt.xticks(fontsize=48.0)
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

                    # matplotlib.use("Qt5Agg")
                    matplotlib.rcParams['savefig.pad_inches'] = 0
                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_subplot(1, 1, 1)
                    ax.xaxis.get_major_formatter()._usetex = False
                    ax.yaxis.get_major_formatter()._usetex = False
                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                    index_particles = []
                    for n in range(n_particle_types):
                        pos = torch.argwhere((T1.squeeze() == n) & (H1[:, 0].squeeze() == 1))
                        pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                        index_particles.append(pos)
                        size = set_size(x, index_particles[n], 10) / 10
                        if 'inv' in style:
                            plt.scatter(to_numpy(X1[index_particles[n], 0]), 1-to_numpy(X1[index_particles[n], 1]),
                                        s=size, color=cmap.color(n))
                        else:
                            plt.scatter(to_numpy(x[index_particles[n], 1]), to_numpy(x[index_particles[n], 2]),
                                        s=size, color=cmap.color(n))
                    dead_cell = np.argwhere(to_numpy(H1[:, 0]) == 0)
                    if len(dead_cell) > 0:
                        if 'inv' in style:
                            plt.scatter(to_numpy(X1[dead_cell[:, 0].squeeze(), 0]),
                                        1-to_numpy(X1[dead_cell[:, 0].squeeze(), 1]), s=2, color='k', alpha=0.5)
                        else:
                            plt.scatter(to_numpy(X1[dead_cell[:, 0].squeeze(), 0]),
                                        to_numpy(X1[dead_cell[:, 0].squeeze(), 1]), s=2, color='k', alpha=0.5)
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    if 'latex' in style:
                        plt.xlabel(r'$x$', fontsize=78)
                        plt.ylabel(r'$y$', fontsize=78)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                    elif 'frame' in style:
                        plt.xlabel('x', fontsize=13)
                        plt.ylabel('y', fontsize=16)
                        plt.xticks(fontsize=16.0)
                        plt.yticks(fontsize=16.0)
                        ax.tick_params(axis='both', which='major', pad=15)
                        plt.text(0, 1.05,
                                 f'frame {it}, {int(n_particles_alive)} alive particles ({int(n_particles_dead)} dead), {edge_index.shape[1]} edges  ',
                                 ha='left', va='top', transform=ax.transAxes, fontsize=16)
                    else:
                        plt.xticks([])
                        plt.yticks([])

                    if 'cell_id' in style:
                        for i, txt in enumerate(to_numpy(N1.squeeze())):
                            if 'inv' in style:
                                plt.text(to_numpy(X1[i, 0]), 1 - to_numpy(X1[i, 1]), 1 + int(to_numpy(N1[i])),
                                         fontsize=8)
                            else:
                                plt.text(to_numpy(X1[i, 0]), to_numpy(X1[i, 1]), 1 + int(to_numpy(N1[i])),
                                     fontsize=8)  # (txt, (to_numpy(X1[i, 0]), to_numpy(X1[i, 1]), 0), fontsize=8)

                    plt.tight_layout()

                    num = f"{it:06}"

                    plt.savefig(f"graphs_data/graphs_{dataset_name}/Fig/Fig_{run}_{num}.tif",
                                dpi=85.35)
                    plt.close()


                fig = plt.figure(figsize=(12, 12))
                ax = fig.add_subplot(2, 2, 1)
                plt.plot(current_loss)
                plt.xlabel('N')
                plt.ylabel('Current_loss')
                ax = fig.add_subplot(2, 2, 2)
                plt.plot(x_len_list, edges_len_list)
                plt.xlabel('Number of particles')
                plt.ylabel('Number of edges')
                ax = fig.add_subplot(2, 2, 3)
                plt.plot(x_len_list)
                plt.xlabel('Number of particles')
                plt.xlabel('Frame')
                ax = fig.add_subplot(2, 2, 4)
                for n in range(n_particle_types):
                    pos = torch.argwhere((T1.squeeze() == n) & (H1[:, 0].squeeze() == 1))
                    # if pos.shape[0] > 1:
                        # sns.kdeplot(to_numpy(AR1[pos].squeeze()), fill=True, color=cmap.color(n), alpha=0.5)
                        # plt.hist(to_numpy(AR1[pos].squeeze()), bins=100, alpha=0.5)
                plt.tight_layout()
                plt.savefig(f"graphs_data/graphs_{dataset_name}/gen_{run}.jpg", dpi=170.7)
                plt.close()


                if 'voronoi' in style:

                    if dimension == 2:

                        vor, vertices_pos, vertices_per_cell, all_points = get_vertices(points=X1, device=device)

                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(1, 1, 1)
                        plt.xticks([])
                        plt.yticks([])

                        voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=1, line_alpha=0.5,
                                        point_size=0)

                        for n in range(n_particle_types):
                            pos = torch.argwhere((T1.squeeze() == n) & (H1[:, 0].squeeze() == 1))
                            if pos.shape[0]>1:
                                pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                                patches = []
                                for i in pos:
                                    cell = vertices_per_cell[i]
                                    vertices = to_numpy(vertices_pos[cell, :])
                                    patches.append(Polygon(vertices, closed=True))
                                if (n==0) & (has_cell_death) & (n_particle_types==3):
                                    pc = PatchCollection(patches, alpha=0.4, facecolors='k')
                                else:
                                    pc = PatchCollection(patches, alpha=0.4, facecolors=cmap.color(n))
                                ax.add_collection(pc)
                            elif pos.shape[0]==1:
                                try:
                                    cell = vertices_per_cell[pos]
                                    vertices = to_numpy(vertices_pos[cell, :])
                                    patches = Polygon(vertices, closed=True)
                                    pc = PatchCollection(patches, alpha=0.4, facecolors=cmap.color(n))
                                    ax.add_collection(pc)
                                except:
                                    pass

                        if 'center' in style:
                            plt.scatter(to_numpy(X1[:, 0]), to_numpy(X1[:, 1]), s=1, c='k')
                            plt.scatter(to_numpy(first_X1[:, 0]), to_numpy(first_X1[:, 1]), s=1, c='r')

                        if 'vertices' in style:
                            plt.scatter(to_numpy(vertices_pos[:, 0]), to_numpy(vertices_pos[:, 1]), s=5, color='k')

                        plt.xlim([-0.05, 1.05])
                        plt.ylim([-0.05, 1.05])

                        if 'cell_id' in style:
                            for i, txt in enumerate(to_numpy(N1.squeeze())):
                                if 'inv' in style:
                                    plt.text(to_numpy(X1[i, 0]), 1 - to_numpy(X1[i, 1]), 1 + int(to_numpy(N1[i])),
                                             fontsize=8)
                                else:
                                    plt.text(to_numpy(X1[i, 0]), to_numpy(X1[i, 1]), 1 + int(to_numpy(N1[i])),
                                         fontsize=8)  # (txt, (to_numpy(X1[i, 0]), to_numpy(X1[i, 1]), 0), fontsize=8)

                        plt.tight_layout()
                        num = f"{it:06}"
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/Fig/Vor_{run}_{num}.tif", dpi=85.35)
                        plt.close()

                    elif dimension == 3:

                        n_particles = len(X1)
                        print(f'frame {it}, {n_particles} particles, {edge_index.shape[1]} edges')
                        vor, vertices_pos, vertices_per_cell, all_points = get_vertices(points=X1, device=device)

                        cells = [[] for i in range(n_particles)]
                        for (l, r), vertices in vor.ridge_dict.items():
                            if l < n_particles:
                                cells[l].append(vor.vertices[vertices])
                            elif r < n_particles:
                                cells[r].append(vor.vertices[vertices])

                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        for n, poly in enumerate(cells):
                            polygon = Poly3DCollection(poly, alpha=0.5,
                                                       facecolors=cmap.color(to_numpy(T1[n]).astype(int)),
                                                       linewidths=0.1, edgecolors='black')
                            ax.add_collection3d(polygon)
                        plt.tight_layout()

                        num = f"{it:06}"
                        plt.savefig(f"graphs_data/graphs_{dataset_name}/Fig/Fig_{run}_{num}.tif", dpi=85.35)
                        plt.close()

        if bSave:
            torch.save(x_list, f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt')
            torch.save(y_list, f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt')
            torch.save(T1_list, f'graphs_data/graphs_{dataset_name}/T1_list_{run}.pt')
            np.savez(f'graphs_data/graphs_{dataset_name}/edge_p_p_list_{run}', *edge_p_p_list)
            if has_inert_model:
                np.savez(f'graphs_data/graphs_{dataset_name}/vertices_pos_list_{run}', *vertices_pos_list)

            torch.save(cycle_length, f'graphs_data/graphs_{dataset_name}/cycle_length.pt')
            torch.save(CL1, f'graphs_data/graphs_{dataset_name}/cycle_length_distrib.pt')
            torch.save(cell_death_rate, f'graphs_data/graphs_{dataset_name}/cell_death_rate.pt')
            torch.save(DR1, f'graphs_data/graphs_{dataset_name}/cell_death_rate_distrib.pt')
            torch.save(model.p, f'graphs_data/graphs_{dataset_name}/model_p.pt')

            if run == 0:
                man_track = to_numpy(man_track)
                pos = np.argwhere(man_track[:, 2] == -1)
                if len(pos) > 0:
                    man_track[pos, 2] = n_frames
                man_track = np.int16(man_track)
                np.savetxt(f'graphs_data/graphs_{dataset_name}/man_track.txt', man_track, fmt="%d", delimiter=" ",
                           newline="\n")

    if bSave:
        run = run + 1

        torch.save(x_list, f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt')
        torch.save(y_list, f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt')
        torch.save(T1_list, f'graphs_data/graphs_{dataset_name}/T1_list_{run}.pt')
        np.savez(f'graphs_data/graphs_{dataset_name}/edge_p_p_list_{run}', *edge_p_p_list)
        if has_inert_model:
            np.savez(f'graphs_data/graphs_{dataset_name}/vertices_pos_list_{run}', *vertices_pos_list)

        torch.save(cycle_length, f'graphs_data/graphs_{dataset_name}/cycle_length.pt')
        torch.save(CL1, f'graphs_data/graphs_{dataset_name}/cycle_length_distrib.pt')
        torch.save(cell_death_rate, f'graphs_data/graphs_{dataset_name}/cell_death_rate.pt')
        torch.save(DR1, f'graphs_data/graphs_{dataset_name}/cell_death_rate_distrib.pt')
        torch.save(model.p, f'graphs_data/graphs_{dataset_name}/model_p.pt')

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def data_generate_cell_from_fluo (config, visualize=True, run_vizualized=0, style='color', erase=False, step=5, alpha=0.2,
                       ratio=1, scenario='none', device=None, bSave=True):
    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(42)

    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    print(f'Generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles_max = simulation_config.n_particles_max
    delta_t = simulation_config.delta_t
    n_frames = simulation_config.n_frames
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset
    marker_size = config.plotting.marker_size
    has_inert_model = simulation_config.cell_inert_model_coeff > 0
    has_cell_death = simulation_config.has_cell_death
    has_cell_division = True

    max_radius_list = []
    edges_len_list = []
    folder = f'./graphs_data/graphs_{dataset_name}/'
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/graphs_{dataset_name}/Fig/', exist_ok=True)
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (f[-3:] != 'Fig') & (f[-2:] != 'GT') & (f != 'p.pt') & (f != 'cycle_length.pt') & (f != 'model_config.json') & (f != 'generation_code.py'):
                os.remove(f)
        files = glob.glob(f'./graphs_data/graphs_{dataset_name}/Fig/*')
        for f in files:
            os.remove(f)


    logging.basicConfig(filename=f'./graphs_data/graphs_{dataset_name}/generator.log', format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(config)

    kill_cell_leaving = False

    for run in range(config.training.n_runs-1):

        torch.cuda.empty_cache()

        model, bc_pos, bc_dpos = choose_model(config=config, device=device)

        n_particles = simulation_config.n_particles

        x_list = []
        y_list = []

        x_len_list = []
        edge_p_p_list = []
        vertices_pos_list = []

        current_loss =[]

        '''
        INITIALIZE PER CELL TYPE VALUES1000
        cycle_length
        final_cell_mass
        cell_death_rate
        mc_slope: scalar used in the function set_mass_coeff

        INITIALIZE PER CELL VALUES
        0 N1 cell index dim=1
        1,2 X1 positions dim=2
        3,4 V1 velocities dim=2
        5 T1 cell type dim=1
        6,7 H1 cell status dim=2  H1[:,0] = cell alive flag, alive : 0 , death : 0 , H1[:,1] = cell division flag, dividing : 1
        8 A1 cell age dim=1
        9 S1 cell stage dim=1  0 = G1 , 1 = S, 2 = G2, 3 = M
        10 M1 cell_mass_distrib mass dim=1 (per node)
        11 R1 cell growth rate dim=1
        12 DR1 cell death rate dim=1
        13 MC1 mass coefficient of the cell (relation between velocity and mass)
        14 AR1 area of the cell
        '''

        path = simulation_config.fluo_path
        files = glob.glob(f"{path}/*.tif")
        frame = 17

        x_list = []
        y_list = []

        for slice in range(70,71):

            x_cell, x_cell_plus, radius, i0 = get_cells_from_fluo(config=config, dimension=dimension, files=files, frame=frame, slice=slice, device=device)
            fluo_width = i0.shape[0]
            target_areas = radius.clone().detach() ** 2 * 3.1411516
            target_areas = target_areas / torch.sum(target_areas)

            X1_ = x_cell_plus.clone().detach()
            X1_.requires_grad = True
            first_X1 = X1_[0:len(x_cell)].clone().detach()
            n_true_cells = len(x_cell)

            optimizer = torch.optim.Adam([X1_], lr=1E-3)

            for k in trange(10):

                vor = Voronoi(to_numpy(X1_) * fluo_width)
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(1, 1, 1)
                plt.imshow(i0)
                voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='w', line_width=1, line_alpha=0.5,
                                point_size=0)
                plt.scatter(to_numpy(x_cell[:, 0])* fluo_width, to_numpy(x_cell[:, 1])* fluo_width, s=to_numpy(target_areas[0:n_true_cells])*2E5, color='g',alpha=0.4, edgecolors='none')
                # plt.scatter(to_numpy(X1_[0:n_true_cells, 0]) * fluo_width, to_numpy(X1_[0:n_true_cells, 1]) * fluo_width, s=10, color='r',alpha=0.5)
                plt.xlim([0, fluo_width])
                plt.ylim([0, fluo_width])
                num = f"{slice:04}"
                fig.savefig(f'graphs_data/graphs_{dataset_name}/Fig/voronoi_{num}_{k}.tif', dpi=280)
                plt.close()

                optimizer.zero_grad()
                vor, vertices_pos, vertices_per_cell, all_points = get_vertices(points=X1_, device=device)
                cc, tri = get_Delaunay(all_points, device)
                distance = torch.sum((vertices_pos[:, None, :].clone().detach() - cc[None, :, :]) ** 2, dim=2)
                result = distance.min(dim=1)
                index = result.indices
                cc = cc[index]

                voronoi_area = get_voronoi_areas(cc, vertices_per_cell, device)
                perimeter = get_voronoi_perimeters(cc, vertices_per_cell, device)

                loss = 5E2 * (target_areas - voronoi_area).norm(2) + 0.2 * torch.sum(perimeter ** 2) + 10 * (first_X1 - X1_[0:n_true_cells]).norm(2)

                # loss = simulation_config.coeff_area * (target_areas - voronoi_area).norm(2)
                # if simulation_config.coeff_perimeter > 0:
                #     loss += simulation_config.coeff_perimeter * torch.sum(perimeter ** 2)

                loss.backward()
                optimizer.step()

            x = torch.cat((x_cell,X1_[n_true_cells:].clone().detach()), 0)
            x = torch.cat((x, voronoi_area[:,None], perimeter[:,None]), 1)
            y = torch.zeros((x.shape[0],1), dtype=torch.float32, device=device)
            y[0:n_true_cells] = 1
            x_list.append(X1_.clone().detach())
            y_list.append(y.clone().detach())

        if bSave:
            torch.save(x_list, f'graphs_data/graphs_{dataset_name}/x_list_{run}.pt')
            torch.save(y_list, f'graphs_data/graphs_{dataset_name}/y_list_{run}.pt')





    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


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
                    if simulation_config.has_cell_division:
                        s_p = 25
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


