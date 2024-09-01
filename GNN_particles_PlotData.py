import matplotlib.pyplot as plt
import imageio
from matplotlib import rc

from ParticleGraph.fitting_models import *
from ParticleGraph.sparsify import *
from ParticleGraph.models.utils import *
from ParticleGraph.models.MLP import *
from ParticleGraph.utils import to_numpy, CustomColorMap
from ParticleGraph.data_loaders import *
from ParticleGraph.utils import bundle_fields

os.environ["PATH"] += os.pathsep + '/usr/local/texlive/2023/bin/x86_64-linux'


def plot_gland(config, config_file, device):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    dataset_name = config.dataset

    print(f'Plot data ... {dataset_name}')

    time_series, global_ids = load_wanglab_salivary_gland(dataset_name, device="cuda:0")

    frame = 180
    frame_data = time_series[frame]

    # IDs are in the range 0, ..., N-1; global ids are stored separately
    print(f"Data fields: {frame_data.node_attrs()}")
    print(f"Number of particles in frame {frame}: {frame_data.num_nodes}")
    print(f"Local ids in frame {frame}: {frame_data.track_id}")
    print(f"Global ids in frame {frame}: {global_ids[frame_data.track_id]}")

    # summarize some of the fields in a particular dataset
    x = bundle_fields(frame_data, "track_id", "pos", "velocity")

    fig, ax = fig_init()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), to_numpy(x[:, 3]), s=25, color='k', alpha=0.05, edgecolors='none')

    # compute the acceleration and a mask to filter out NaN values
    acceleration, mask = time_series.compute_derivative("velocity", id_name="track_id")
    Y = acceleration[frame]
    Y = Y[mask[frame], :]
    print(f"NaNs in sanitized acceleration: {torch.isnan(Y).sum()}")

    # Sanity-check one to one correspondence between X and Y
    #   pred = GNN(X)
    #   loss = pred[mask] - Y[mask]

    # stack all the accelerations / masks
    acceleration = torch.vstack(acceleration)
    mask = torch.hstack(mask)
    std = torch.std(acceleration[mask, :], dim=0)

    # get velocity for all time steps
    velocity = torch.vstack([frame.velocity for frame in time_series])

    # a TimeSeries object can be sliced like a list
    every_second_frame = time_series[::2]
    first_ten_frames = time_series[:10]
    last_ten_frames_reversed = time_series[-1:-11:-1]

def plot_celegans(config, config_file, device):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Plot data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    delta_t = simulation_config.delta_t
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    n_runs = train_config.n_runs
    dimension = 3

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))

    os.makedirs(os.path.join(log_dir, 'generated_bw'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'generated_voronoi'), exist_ok=True)


    files = glob.glob(f"./{log_dir}/generated_bw/*")
    for f in files:
        os.remove(f)
    files = glob.glob(f"./{log_dir}/generated_voronoi/*")
    for f in files:
        os.remove(f)

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    print('Load data ...')

    time_series, cell_info = load_celegans_gene_data(dataset_name, device=device)

    # The info about the cells is stored in a pandas DataFrame
    print(f"Number of cells: {len(cell_info)}")
    print(cell_info.describe())

    cell_name = cell_info.index[5]
    cell_type = cell_info['type'].iloc[5]
    print(f"First few gene names: {cell_info.index[:6]}")
    print(f"Cell type for {cell_name}: {cell_type}")

    # Since the location and gene data are acquired at different time intervals, the time series only contains
    # the time in which both data are available (gene data is linearly interpolated)
    print(f"Number of time points: {len(time_series)}")
    print(f"Time points: {time_series.time}")

    # The data objects contain the fields 'pos' and 'gene_cpm' and their derivatives ('velocity' and 'd_gene_cpm')
    time_point = time_series[0]
    print(f"Time point fields: {time_point.node_attrs()}")
    print(f"Number of cells in time point 0: {time_point.num_nodes}")

    print(f"Position shape: {time_point.pos.shape}")
    print(f"Velocity shape: {time_point.velocity.shape}")
    print(f"Gene expression shape: {time_point.gene_cpm.shape}")
    print(f"Gene expression derivative shape: {time_point.d_gene_cpm.shape}")

    data = time_series[50]
    x = bundle_fields(data, "pos").clone().detach()


    n_backgrounds = 1000
    count = 1
    intermediate_count = 0
    distance_threshold = 50
    while count < n_backgrounds+1:

        new_pos = torch.rand(1, dimension, device=device)
        new_pos[:, 0] = new_pos[:, 0] * 20 - 10
        new_pos[:, 1] = new_pos[:, 1] * 20 - 10
        new_pos[:, 2] *= 175

        distance = torch.sum((x[:, None, 0:3] - new_pos[None, :, :]) ** 2, dim=2)
        if torch.all(distance > distance_threshold ** 2):
            x = torch.cat((x, new_pos), 0)
            count += 1
        intermediate_count += 1
        if intermediate_count > 100:
            distance_threshold = distance_threshold * 0.99
            intermediate_count = 0

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', box_aspect=[1, 1, 1])
    ax.scatter(to_numpy(x[-1000:, 0]), to_numpy(x[-1000:, 1]), to_numpy(x[-1000:, 2]), c="r", marker="o", s=1)
    ax.scatter(to_numpy(x[:321, 0]), to_numpy(x[:321, 1]), to_numpy(x[:321, 2]), c="k", marker="o", s=10)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 175)

    fig = plt.figure()
    plt.scatter(to_numpy(x[-1000:, 0]), to_numpy(x[-1000:, 1]), c="r", marker="o")
    ax.scatter(to_numpy(x[:321, 0]), to_numpy(x[:321, 1]), to_numpy(x[:321, 2]), c="k", marker="o")

def plot_generated_agents(config, config_file, device):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    print(f'Plot data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
    delta_t = simulation_config.delta_t
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames
    n_runs = train_config.n_runs

    l_dir = os.path.join('.', 'log')
    log_dir = os.path.join(l_dir, 'try_{}'.format(config_file))

    os.makedirs(os.path.join(log_dir, 'generated_bw'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'generated_velocity'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'generated_internal'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'generated_state'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'generated_reversal_timer'), exist_ok=True)

    files = glob.glob(f"./{log_dir}/generated_bw/*")
    for f in files:
        os.remove(f)
    files = glob.glob(f"./{log_dir}/generated_velocity/*")
    for f in files:
        os.remove(f)
    files = glob.glob(f"./{log_dir}/generated_internal/*")
    for f in files:
        os.remove(f)
    files = glob.glob(f"./{log_dir}/generated_state/*")
    for f in files:
        os.remove(f)
    files = glob.glob(f"./{log_dir}/generated_reversal_timer/*")
    for f in files:
        os.remove(f)

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    print('Load data ...')

    time_series, signal = load_agent_data(dataset_name, device=device)

    velocities = [t.velocity for t in time_series]
    velocities.pop(0)  # the first element is always NaN
    velocities = torch.stack(velocities)
    if torch.any(torch.isnan(velocities)):
        raise ValueError('Discovered NaN in velocities. Aborting.')
    velocities = bc_dpos(velocities)

    vnorm = torch.std(velocities[:, :, 0])
    ynorm = vnorm

    positions = torch.stack([t.pos for t in time_series])
    min = torch.min(positions[:, :, 0])
    max = torch.max(positions[:, :, 0])
    mean = torch.mean(positions[:, :, 0])
    std = torch.std(positions[:, :, 0])
    print(f"min: {min}, max: {max}, mean: {mean}, std: {std}")

    n_particles = config.simulation.n_particles
    print(f'N particles: {n_particles}')

    if os.path.exists(f'./log/try_{config_file}/edge_p_p_list.npz'):
        print('Load list of edges index ...')
        edge_p_p_list = np.load(f'./log/try_{config_file}/edge_p_p_list.npz')
    else:
        print('Create list of edges index ...')
        edge_p_p_list = []
        for k in trange(n_frames):
            time_point = time_series[k]
            x = bundle_fields(time_point, "pos", "velocity", "internal", "state", "reversal_timer").clone().detach()
            x = torch.column_stack((torch.arange(0, n_particles, device=device), x))

            nbrs = NearestNeighbors(n_neighbors=simulation_config.n_neighbors, algorithm='auto').fit(to_numpy(x[:, 1:dimension + 1]))
            distances, indices = nbrs.kneighbors(to_numpy(x[:, 1:dimension + 1]))
            edge_index = []
            for i in range(indices.shape[0]):
                for j in range(1, indices.shape[1]):  # Start from 1 to avoid self-loop
                    edge_index.append((i, indices[i, j]))
            edge_index = np.array(edge_index)
            edge_index = torch.tensor(edge_index, device=device).t().contiguous()
            edge_p_p_list.append(to_numpy(edge_index))
        np.savez(f'./log/try_{config_file}/edge_p_p_list', *edge_p_p_list)

    for k in trange(1,n_frames):

        time_point = time_series[k]
        x = bundle_fields(time_point, "pos", "velocity", "internal", "state", "reversal_timer").clone().detach()
        x = torch.column_stack((torch.arange(0,n_particles, device=device),x))
        x [:, 1:5] = x[:, 1:5]/1000
        x[:,3:5] = bc_dpos(x[:,3:5])

        edges = edge_p_p_list[f'arr_{k}']
        edges = torch.tensor(edges, dtype=torch.int64, device=device)
        dataset = data.Data(x=x[:, :], edge_index=edges)

        fig, ax = fig_init()
        plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), s=1, alpha=0.1,c='k')
        plt.xticks(fontsize=16.0)
        plt.yticks(fontsize=16.0)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/generated_bw/Fig_{k}.tif", dpi=87)
        plt.close()

        v = to_numpy(torch.norm(x[:, 3:5], dim=1))
        fig, ax = fig_init()
        plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), c=v, s=1, alpha=1.0, cmap='viridis',vmin=0, vmax=vnorm/20)
        plt.xticks(fontsize=16.0)
        plt.yticks(fontsize=16.0)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/generated_velocity/Fig_{k}.tif", dpi=87)
        plt.close()

        internal = to_numpy(torch.norm(x[:, 5:6], dim=1))
        fig, ax = fig_init()
        plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), c=internal, s=1, alpha=1.0, cmap='viridis',vmin=0, vmax=0.5)
        plt.xticks(fontsize=16.0)
        plt.yticks(fontsize=16.0)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/generated_internal/Fig_{k}.tif", dpi=87)
        plt.close()

        state = to_numpy(torch.norm(x[:, 6:7], dim=1))
        fig, ax = fig_init()
        plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), c=state, s=1, alpha=1.0, cmap='viridis',vmin=0, vmax=1)
        plt.xticks(fontsize=16.0)
        plt.yticks(fontsize=16.0)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/generated_state/Fig_{k}.tif", dpi=87)
        plt.close()

        reversal_timer = to_numpy(torch.norm(x[:, 7:8], dim=1))
        fig, ax = fig_init()
        plt.scatter(to_numpy(x[:, 2]), to_numpy(x[:, 1]), c=reversal_timer, s=1, alpha=1.0, cmap='viridis',vmin=0, vmax=125)
        plt.xticks(fontsize=16.0)
        plt.yticks(fontsize=16.0)
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/generated_reversal_timer/Fig_{k}.tif", dpi=87)
        plt.close()

def plot_gravity_solar_system(config_file, epoch_list, log_dir, logger, device):
    config_file = 'gravity_solar_system'
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')

    dataset_name = config.dataset
    embedding_cluster = EmbeddingCluster(config)

    cmap = CustomColorMap(config=config)
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    n_particle_types = config.simulation.n_particle_types
    n_particles = config.simulation.n_particles
    n_runs = config.training.n_runs
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius

    time.sleep(0.5)

    x_list = []
    y_list = []
    print('Load data ...')
    time.sleep(1)
    x_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/x_list_0.pt', map_location=device))
    y_list.append(torch.load(f'graphs_data/graphs_{dataset_name}/y_list_0.pt', map_location=device))
    vnorm = torch.load(os.path.join(log_dir, 'vnorm.pt'), map_location=device)
    ynorm = torch.load(os.path.join(log_dir, 'ynorm.pt'), map_location=device)
    x = x_list[0][0].clone().detach()
    index_particles = get_index_particles(x, n_particle_types, dimension)
    type_list = get_type_list(x, dimension)

    model, bc_pos, bc_dpos = choose_training_model(config, device)

    net = f"./log/try_{config_file}/models/best_model_with_{n_runs - 1}_graphs_2.pt"
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    plt.rcParams['text.usetex'] = True
    rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    # matplotlib.use("Qt5Agg")

    fig = plt.figure(figsize=(10.5, 9.6))
    plt.ion()
    ax = fig.add_subplot(3, 3, 1)
    embedding = plot_embedding('a)', model.a, 1, index_particles, n_particles, n_particle_types, 20, '$10^6$', fig, ax,
                               cmap, device)

    ax = fig.add_subplot(3, 3, 2)
    rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
    func_list = plot_function(True, 'b)', config.graph_model.particle_model_name, model.lin_edge,
                              model.a, 1, to_numpy(x[:, 5]).astype(int), rr, max_radius, ynorm, index_particles,
                              n_particles, n_particle_types, 20, '$10^6$', fig, ax, cmap, device)

    ax = fig.add_subplot(3, 3, 3)

    it = 2000
    x0 = x_list[0][it].clone().detach()
    y0 = y_list[0][it].clone().detach()
    x = x_list[0][it].clone().detach()
    distance = torch.sum(bc_dpos(x[:, None, 1:3] - x[None, :, 1:3]) ** 2, dim=2)
    t = torch.Tensor([max_radius ** 2])  # threshold
    adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)) * 1.0
    edge_index = adj_t.nonzero().t().contiguous()
    dataset = data.Data(x=x, edge_index=edge_index)

    with torch.no_grad():
        y = model(dataset, data_id=1, training=False, vnorm=vnorm,
                  phi=torch.zeros(1, device=device))  # acceleration estimation
    y = y * ynorm

    proj_interaction, new_labels, n_clusters = plot_umap('b)', func_list, log_dir, 500, index_particles,
                                                         n_particles, n_particle_types, embedding_cluster, 20, '$10^6$',
                                                         fig, ax, cmap, device)

    ax = fig.add_subplot(3, 3, 3)
    accuracy = plot_confusion_matrix('c)', to_numpy(x[:, 5:6]), new_labels, n_particle_types, 20, '$10^6$', fig, ax)
    plt.tight_layout()

    model_a_ = model.a.clone().detach()
    model_a_ = torch.reshape(model_a_, (model_a_.shape[0] * model_a_.shape[1], model_a_.shape[2]))
    for k in range(n_clusters):
        pos = np.argwhere(new_labels == k).squeeze().astype(int)
        temp = model_a_[pos, :].clone().detach()
        model_a_[pos, :] = torch.median(temp, dim=0).values.repeat((len(pos), 1))
    with torch.no_grad():
        for n in range(model.a.shape[0]):
            model.a[n] = model_a_
    embedding, embedding_particle = get_embedding(model.a, 1)

    ax = fig.add_subplot(3, 3, 4)
    plt.text(-0.25, 1.1, f'd)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.title(r'Clustered particle embedding', fontsize=12)
    for n in range(n_particle_types):
        pos = np.argwhere(new_labels == n).squeeze().astype(int)
        plt.scatter(embedding[pos[0], 0], embedding[pos[0], 1], color=cmap.color(n), s=6)
    plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=12)
    plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=12)
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)
    plt.text(.05, .94, f'e: 20 it: $10^6$', ha='left', va='top', transform=ax.transAxes, fontsize=10)

    ax = fig.add_subplot(3, 3, 5)
    print('5')
    plt.text(-0.25, 1.1, f'e)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.title(r'Interaction functions (model)', fontsize=12)
    func_list = []
    for n in range(n_particle_types):
        pos = np.argwhere(new_labels == n).squeeze().astype(int)
        embedding = model.a[0, pos[0], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                 rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                 0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        with torch.no_grad():
            func = model.lin_edge(in_features.float())
        func = func[:, 0]
        func_list.append(func)
        plt.plot(to_numpy(rr),
                 to_numpy(func) * to_numpy(ynorm),
                 color=cmap.color(n), linewidth=1)
    plt.xlabel(r'$d_{ij}$', fontsize=12)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij}$', fontsize=12)
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)
    plt.xlim([0, 0.02])
    plt.ylim([0, 0.5E6])
    plt.text(.05, .94, f'e: 20 it: $10^6$', ha='left', va='top', transform=ax.transAxes, fontsize=10)

    ax = fig.add_subplot(3, 3, 6)
    print('6')
    plt.text(-0.25, 1.1, f'k)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.title(r'Interaction functions (true)', fontsize=12)
    p = np.linspace(0.5, 5, n_particle_types)
    p = torch.tensor(p, device=device)
    for n in range(n_particle_types - 1, -1, -1):
        plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n), linewidth=1)
    plt.xlabel(r'$d_{ij}$', fontsize=12)
    plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij}$', fontsize=12)
    plt.xticks(fontsize=10.0)
    plt.yticks(fontsize=10.0)
    plt.xlim([0, 0.02])
    plt.ylim([0, 0.5E6])

    plot_list = []
    for n in range(n_particle_types):
        pos = np.argwhere(new_labels == n).squeeze().astype(int)
        embedding = model.a[0, pos[0], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        if config.graph_model.prediction == '2nd_derivative':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        else:
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding), dim=1)
        with torch.no_grad():
            pred = model.lin_edge(in_features.float())
        pred = pred[:, 0]
        plot_list.append(pred * ynorm)
    p = np.linspace(0.5, 5, n_particle_types)
    popt_list = []
    for n in range(n_particle_types):
        popt, pcov = curve_fit(power_model, to_numpy(rr), to_numpy(plot_list[n]))
        popt_list.append(popt)
    popt_list = np.array(popt_list)

    ax = fig.add_subplot(3, 3, 7)
    print('7')
    plt.text(-0.25, 1.1, f'g)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    x_data = p
    y_data = popt_list[:, 0]
    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
    plt.plot(p, linear_model(x_data, lin_fit[0], lin_fit[1]), color='r', linewidth=0.5)
    plt.scatter(p, popt_list[:, 0], color='k', s=20)
    plt.title(r'Learned masses', fontsize=64)
    plt.xlabel(r'True mass ', fontsize=64)
    plt.ylabel(r'Predicted mass ', fontsize=64)
    plt.xlim([0, 5.5])
    plt.ylim([0, 5.5])
    plt.text(0.5, 5, f"Slope: {np.round(lin_fit[0], 2)}", fontsize=10)
    residuals = y_data - linear_model(x_data, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    plt.text(0.5, 4.5, f"$R^2$: {np.round(r_squared, 3)}", fontsize=10)

    ax = fig.add_subplot(3, 3, 8)
    print('8')
    plt.text(-0.25, 1.1, f'h)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
    plt.scatter(p, -popt_list[:, 1], color='k', s=20)
    plt.xlim([0, 5.5])
    plt.ylim([-4, 0])
    plt.title(r'Learned exponent', fontsize=12)
    plt.xlabel(r'True mass ', fontsize=12)
    plt.ylabel(r'Exponent fit ', fontsize=12)
    plt.text(0.5, -0.5, f"Exponent: {np.round(np.mean(-popt_list[:, 1]), 3)}+/-{np.round(np.std(popt_list[:, 1]), 3)}",
             fontsize=10)

    # find last image file in logdir
    ax = fig.add_subplot(3, 3, 9)
    files = glob.glob(os.path.join(log_dir, 'tmp_recons/Fig*.tif'))
    files.sort(key=os.path.getmtime)
    if len(files) > 0:
        last_file = files[-1]
        # load image file with imageio
        image = imageio.imread(last_file)
        print('12')
        plt.text(-0.25, 1.1, f'l)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
        plt.title(r'Rollout inference (frame 1000)', fontsize=12)
        plt.imshow(image)
        # rmove xtick
        plt.xticks([])
        plt.yticks([])

    time.sleep(1)
    plt.tight_layout()
    # plt.savefig('Fig3.pdf', format="pdf", dpi=300)
    plt.savefig('Fig3.jpg', dpi=300)
    plt.close()


# if __name__ == '__main__':
#
#   matplotlib.use("Qt5Agg")
#
#   p = (np.random.random((100, 3)))*10.0-5
#   p[:, 1] += 2.5
#   pos = jp.array(p)
#   c = np.random.random((1000, 3))
#   color = jp.array(c)
#   balls = Balls(pos, color)
#
#   # show_slice(partial(balls_sdf, balls), z=0.0)
#
#   w, h = 640, 640
#   pos0 = jp.float32([5,30,35])
#   pos0 = jp.float32([2.5, 15, 17.5])
#   ray_dir = camera_rays(-pos0, view_size=(w, h))
#   sdf = partial(scene_sdf, balls)
#   hit_pos = jax.vmap(partial(raycast, sdf, pos0))(ray_dir)
#   # pl.imshow(hit_pos.reshape(h, w, 3)%1.0)
#   raw_normal = jax.vmap(jax.grad(sdf))(hit_pos)
#   # pl.imshow(raw_normal.reshape(h, w, 3))
#   light_dir = normalize(jp.array([1.1, 1.0, 0.2]))
#   shadow = jax.vmap(partial(cast_shadow, sdf, light_dir))(hit_pos)
#   # pl.imshow(shadow.reshape(h, w))
#   f = partial(shade_f, jp.ones(3), light_dir=light_dir)
#   frame = jax.vmap(f)(shadow, raw_normal, ray_dir)
#   frame = frame ** (1.0 / 2.2)  # gamma correction
#   # pl.imshow(frame.reshape(h, w, 3))
#   color_sdf = partial(scene_sdf, balls, with_color=True)
#   _, surf_color = jax.vmap(color_sdf)(hit_pos)
#   f = partial(shade_f, light_dir=light_dir)
#   frame = jax.vmap(f)(surf_color, shadow, raw_normal, ray_dir)
#   frame = frame**(1.0/2.2)  # gamma correction
#   pl.figure(figsize=(8, 8))
#   pl.imshow(frame.reshape(h, w, 3))



if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(' ')
    print(f'device {device}')
    print(' ')

    matplotlib.use("Qt5Agg")

    # config_file = "celegans"
    # config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    # plot_celegans(config, config_file, device)


    # config_file = "agents"
    # config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    # plot_generated_agents(config, config_file, device)


    config_file = "gland"
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    plot_gland(config, config_file, device)


    # f_list = ['supp13']
    # for f in f_list:
    #     config_list,epoch_list = get_figures(f)





