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


if __name__ == '__main__':

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(' ')
    print(f'device {device}')
    print(' ')

    matplotlib.use("Qt5Agg")

    config_file = "arbitrary_3"
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    plot_gland(config, config_file, device)

    # f_list = ['supp13']
    # for f in f_list:
    #     config_list,epoch_list = get_figures(f)





