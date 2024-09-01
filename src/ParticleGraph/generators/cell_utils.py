import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d,  Delaunay
import torch
from ParticleGraph.utils import to_numpy
import math
import torch_geometric.data as data
import matplotlib.pyplot as plt

from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tifffile import imread, imsave
import glob
from skimage.measure import regionprops

def init_cell_range(config, device, scenario="None"):
    simulation_config = config.simulation
    n_particles = simulation_config.n_particles
    n_particle_types = simulation_config.n_particle_types

    ##### defines all variables for the cell model, per type of cell: dimension = n_particle_types

    if config.simulation.cell_cycle_length != [-1]:
        cycle_length = torch.tensor(config.simulation.cell_cycle_length, device=device)
    else:
        cycle_length = torch.clamp(torch.abs(torch.ones(n_particle_types, 1, device=device) * 250 + torch.randn(n_particle_types, 1, device=device) * 50), min=100, max=700).squeeze()

    if config.simulation.final_cell_mass != [-1]:
        final_cell_mass = torch.tensor(config.simulation.final_cell_mass, device=device)
    else:
        final_cell_mass = torch.clamp(torch.abs(
            torch.ones(n_particle_types, 1, device=device) * 250 + torch.randn(n_particle_types, 1,
                                                                               device=device) * 25), min=200,
                                      max=500).flatten()

    if config.simulation.cell_death_rate != [-1]:
        cell_death_rate = torch.tensor(config.simulation.cell_death_rate, device=device)
    else:
        cell_death_rate = torch.zeros((n_particles, 1), device=device)

    if config.simulation.mc_slope != [-1]:
        mc_slope = torch.tensor(config.simulation.mc_slope, device=device)
    else:
        mc_slope = torch.clamp(torch.randn(n_particle_types, 1, device=device) * 30, min=-30, max=30).flatten()

    if config.simulation.cell_area != [-1]:
        cell_area = torch.tensor(config.simulation.cell_area, device=device)
    else:
        cell_area = torch.clamp(torch.abs(torch.ones(n_particle_types, 1, device=device) * 0.0015 + torch.randn(n_particle_types, 1, device=device) * 0.0010), min=0.0005, max=0.0025).squeeze()

    return cycle_length, final_cell_mass, cell_death_rate, mc_slope, cell_area

def init_cells(config, cycle_length, final_cell_mass, cell_death_rate, mc_slope, cell_area, bc_pos, bc_dpos, dimension, device):
    simulation_config = config.simulation
    n_particles = simulation_config.n_particles
    n_particle_types = simulation_config.n_particle_types
    dimension = simulation_config.dimension

    dpos_init = simulation_config.dpos_init

    if (simulation_config.boundary == 'periodic'):  # | (simulation_config.dimension == 3):

        pos = torch.rand(1, dimension, device=device)
        count = 1
        intermediate_count = 0
        distance_threshold = 0.025
        while count < n_particles:
            new_pos = torch.rand(1, dimension, device=device)
            distance = torch.sum(bc_dpos(pos[:, None, :] - new_pos[None, :, :]) ** 2, dim=2)
            if torch.all(distance > distance_threshold**2):
                pos = torch.cat((pos, new_pos), 0)
                count += 1
            intermediate_count += 1
            if intermediate_count > 100:
                distance_threshold = distance_threshold * 0.99
                intermediate_count = 0

    else:
        pos = torch.randn(n_particles, dimension, device=device) * 0.5

    ###### specify all variables per cell, dimension = n_particles

    # specify position
    dpos = dpos_init * torch.randn((n_particles, dimension), device=device)
    dpos = torch.clamp(dpos, min=-torch.std(dpos), max=+torch.std(dpos))
    # specify type
    if config.simulation.cell_type_map is not None:
        i0 = imread(f'graphs_data/{config.simulation.cell_type_map}')
        type_values = np.unique(i0)
        i0_ = np.zeros_like(i0)
        for n, pixel_values in enumerate(type_values):
            i0_[i0 == pixel_values] = n
        type = i0_[255-(to_numpy(pos[:, 1]) * 255).astype(int), (to_numpy(pos[:, 0]) * 255).astype(int)].astype(int)
        type = torch.tensor(type, device=device)
        type = torch.clamp(type, min=0, max=n_particle_types - 1)
    else:
        type = torch.zeros(int(n_particles / n_particle_types), device=device)
        for n in range(1, n_particle_types):
            type = torch.cat((type, n * torch.ones(int(n_particles / n_particle_types), device=device)), 0)
        if (simulation_config.params == 'continuous') | (
                config.simulation.non_discrete_level > 0):  # TODO: params is a list[list[float]]; this can never happen?
            type = torch.tensor(np.arange(n_particles), device=device)
    # specify cell status dim=2  H1[:,0] = cell alive flag, alive : 0 , death : 0 , H1[:,1] = cell division flag, dividing : 1
    status = torch.ones(n_particles, 2, device=device)
    status[:, 1] = 0

    cycle_length_distrib = cycle_length[to_numpy(type)] * (
                torch.ones(n_particles, device=device) + 0.05 * torch.randn(n_particles, device=device))
    cycle_length_distrib = cycle_length_distrib[:, None]

    mc_slope_distrib = mc_slope[to_numpy(
        type), None]  # * (torch.ones(n_particles, device=device) + 0.05 * torch.randn(n_particles, device=device))

    cell_age = torch.rand(n_particles, device=device)
    cell_age = cell_age * cycle_length[to_numpy(type)].squeeze()
    cell_age = cell_age[:, None]
    cell_stage = update_cell_cycle_stage(cell_age, cycle_length, type, device)

    growth_rate = final_cell_mass / (2 * cycle_length)
    growth_rate_distrib = growth_rate[to_numpy(type)].squeeze()[:, None]

    cell_mass_distrib = (growth_rate_distrib * cell_age) + (final_cell_mass[to_numpy(type), None] / 2)

    cell_death_rate_distrib = (cell_death_rate[to_numpy(type)].squeeze() * (
                torch.ones(n_particles, device=device) + 0.05 * torch.randn(n_particles, device=device))) / 100
    cell_death_rate_distrib = cell_death_rate_distrib[:, None]

    cell_area_distrib = cell_area[to_numpy(type)].squeeze()[:, None]

    particle_id = torch.arange(n_particles, device=device)
    particle_id = particle_id[:, None]
    type = type[:, None]

    perimeter = torch.zeros((n_particles,1), device=device)

    return particle_id, pos, dpos, type, status, cell_age, cell_stage, cell_mass_distrib, growth_rate_distrib, cycle_length_distrib, cell_death_rate_distrib, mc_slope_distrib, cell_area_distrib, perimeter

def get_cells_from_fluo(config, dimension, files, frame, slice, device):
    simulation_config = config.simulation
    n_particles = simulation_config.n_particles
    n_particle_types = simulation_config.n_particle_types
    dimension = simulation_config.dimension
    fluo_method = simulation_config.fluo_method

    i0 = imread(files[frame])
    if slice == -1:
        i0 = i0
    else:
        i0 = i0[slice,:,:]
    fluo_width = i0.shape[0]

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(i0)

    regions = regionprops(i0)
    y_ = []
    x_ = []
    r_ = []
    list_mask = []
    for reg in regions:
        y_.append(reg.centroid[0])
        x_.append(reg.centroid[1])
        r_.append(reg.equivalent_diameter / 2)

    n_cells = len(regions)
    print(f'{n_cells} cells')
    x = np.array(x_)[:, None]
    y = np.array(y_)[:, None]
    r = np.array(r_)[:, None]

    # fig = plt.figure(figsize=(10, 10))
    # plt.imshow(i0 * 0, cmap='grey')
    # plt.scatter(x, y, c='g', s=r ** 2, edgecolors='none')

    x = np.concatenate((x, y, r), axis=1)
    x = torch.tensor(x, device=device)

    match fluo_method:
        case 'add':
            n_particles = 6000
            count = 1
            intermediate_count = 0
            distance_threshold = 10
            r_mean = torch.mean(x[:, 2])
            while count < n_particles:
                new_pos = torch.rand(1, 3, device=device) * fluo_width
                new_pos[0, 2] = r_mean
                distance = torch.sum((x[:, None, 0:2] - new_pos[None, :, 0:2]) ** 2, dim=2)
                if torch.all(distance > distance_threshold ** 2):
                    x = torch.cat((x, new_pos), 0)
                    count += 1
                intermediate_count += 1
                if intermediate_count > 100:
                    distance_threshold = distance_threshold * 0.99
                    intermediate_count = 0

    x_cell = x[0:len(regions), 0:2] / fluo_width
    x_cell_plus = x[:,0:2] / fluo_width
    radius = x[:,2] / fluo_width


    return x_cell, x_cell_plus, radius, i0

def update_cell_cycle_stage(cell_age, cycle_length, type_list, device):
    g1 = 0.46
    s = 0.33
    g2 = 0.17
    m = 0.04

    G1 = (g1 * cycle_length).squeeze()
    S = ((g1 + s) * cycle_length).squeeze()
    G2 = ((g1 + s + g2) * cycle_length).squeeze()
    M = ((g1 + s + g2 + m) * cycle_length).squeeze()

    cell_age = cell_age.squeeze()

    cell_stage = torch.zeros(len(cell_age), device=device)
    for i in range(len(cell_age)):
        curr = cell_age[i]

        if curr <= G1[int(type_list[i])]:
            cell_stage[i] = 0
        elif curr <= S[int(type_list[i])]:
            cell_stage[i] = 1
        elif curr <= G2[int(type_list[i])]:
            cell_stage[i] = 2
        else:
            cell_stage[i] = 3

    return cell_stage[:, None]

def get_vertices(points=[], device=[]):

    all_points = points
    if points.shape[1] == 3:   # has 3D
        v_list = [[-1, -1, 1], [-1, 0, 1], [-1, 1, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1], [1, -1, 1], [0, -1, 1], [0, 0, 1],
                  [-1, -1, 0], [-1, 0, 0], [-1, 1, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0], [1, -1, 0], [0, -1, 0],
                  [-1, -1, -1], [-1, 0, -1], [-1, 1, -1], [0, 1, -1], [1, 1, -1], [1, 0, -1], [1, -1, -1], [0, -1, -1], [0, 0, -1]]
    else:
        v_list = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1]]
    v_list = torch.tensor(v_list, device=device)
    for n in range(len(v_list)):
        all_points = torch.concatenate((all_points, points + v_list[n]), axis=0)

    if points.shape[1] == 3:
        pos = torch.argwhere((all_points[:, 0] > -0.05) & (all_points[:, 0] < 1.05) & (all_points[:, 1] > -0.05) & (
                    all_points[:, 1] < 1.05) & (all_points[:, 2] > -0.05) & (all_points[:, 2] < 1.05))
    else:
        pos = torch.argwhere ((all_points[:,0] >-0.05) & (all_points[:,0] <1.05) & (all_points[:,1] >-0.05) & (all_points[:,1] <1.05))
    all_points = all_points[pos].squeeze()

    vor = Voronoi(to_numpy(all_points))

    fig = plt.figure(figsize=(10, 10))
    # voronoi_plot_2d(vor, ax=fig.gca(), show_vertices=False, line_colors='black', line_width=1, line_alpha=0.5)
    # plt.scatter(to_numpy(points[:, 0]), to_numpy(points[:, 1]), s=30, color='red')

    # vertices_index collect all vertices index of regions of interest
    vertices_per_cell = []
    for n in range(len(points)):
        if n == 0:
            vertices_index = vor.regions[vor.point_region[0].copy()]
        else:
            vertices_index = np.concatenate((vertices_index, vor.regions[vor.point_region[n]]), axis=0)
        vertices_per_cell.append((vor.regions[vor.point_region[n]].copy()))

    vertices = []
    map = {}
    count = 0
    for i in range(len(vertices_per_cell)):
        for j in range(len(vertices_per_cell[i])):
            if vertices_per_cell[i][j] in map:
                vertices_per_cell[i][j] = map[vertices_per_cell[i][j]]
            else:
                map[vertices_per_cell[i][j]] = count
                vertices.append(vor.vertices[vertices_per_cell[i][j]])
                vertices_per_cell[i][j] = map[vertices_per_cell[i][j]]
                count += 1
    vertices_pos = np.array(vertices)
    vertices_pos = torch.tensor(vertices_pos, device=device)
    vertices_pos = vertices_pos.to(dtype=torch.float32)

    return vor, vertices_pos, vertices_per_cell, all_points

def get_Delaunay(points=[], device=[]):

    tri = Delaunay(to_numpy(points))  # Compute Delaunay triangulation

    p = points[tri.simplices]  # Triangle vertices

    # Triangle vertices
    A = p[:,0,:].T
    B = p[:,1,:].T
    C = p[:,2,:].T

    # fig = plt.figure()
    # plt.scatter(to_numpy(A[0, 100]), to_numpy(A[1,100]), s=10, color='blue')
    # plt.scatter(to_numpy(B[0, 100]), to_numpy(B[1,100]), s=10, color='blue')
    # plt.scatter(to_numpy(C[0, 100]), to_numpy(C[1,100]), s=10, color='blue')

    # Compute circumcenters (cc)
    a = A - C
    b = B - C

    cc = cross2(sq2(a) * b - sq2(b) * a, a, b) / (2 * ncross2(a, b) + 1E-16) + C

    # plt.scatter(to_numpy(cc[0, 100]), to_numpy(cc[1,100]), s=10, color='red')

    cc = cc.t()

    return cc, tri.simplices

def dot2(u, v):
    return u[0]*v[0] + u[1]*v[1]

def cross2(u, v, w):
    """u x (v x w)"""
    return dot2(u, w)*v - dot2(u, v)*w

def ncross2(u, v):
    """|| u x v ||^2"""
    return sq2(u)*sq2(v) - dot2(u, v)**2

def sq2(u):
    return dot2(u, u)

def get_voronoi_areas(vertices_pos, vertices_per_cell, device):

    centroids = get_voronoi_centroids(vertices_pos, vertices_per_cell, device)
    areas = []
    
    for i in range(len(vertices_per_cell)):
        v_list = vertices_per_cell[i]
        per_cell = 0
        for v in range(-1, len(v_list)-1):
            vert1 = vertices_pos[v_list[v]]-centroids[i]
            vert2 = vertices_pos[v_list[v+1]]-centroids[i]
            cross_product = vert1[0] * vert2[1] - vert1[1] * vert2[0]
            per_cell += torch.abs(cross_product)/2

        areas.append(per_cell)

    areas = torch.stack(areas)

    return areas

def get_voronoi_perimeters(vertices_pos, vertices_per_cell, device):
    perimeters = []
    for v_list in vertices_per_cell:

        per_cell = 0
        for v in range(-1, len(v_list)-1):
            v1 = vertices_pos[v_list[v]]
            v2 = vertices_pos[v_list[v+1]]
            per_cell += torch.dist(v1, v2)

        perimeters.append(per_cell)

    perimeters = torch.stack(perimeters)

    return perimeters

def get_voronoi_lengths(vertices_pos, vertices_per_cell, device):

    lengths = []
    for v_list in vertices_per_cell:

        per_cell = []
        for v in range(-1, len(v_list)-1):
            v1 = vertices_pos[v_list[v]]
            v2 = vertices_pos[v_list[v+1]]
            per_cell.append(torch.dist(v1, v2))

        lengths.append(per_cell)

    return lengths

def get_voronoi_centroids(vertices_pos, vertices_per_cell, device):

    centroids = []
    for v_list in vertices_per_cell:
        centroids.append(torch.mean(vertices_pos[v_list],dim=0))

    centroids = torch.stack(centroids)

    return centroids

def cell_energy(voronoi_area, voronoi_perimeter, voronoi_lengths, device):

    energy = []
    return energy












# fig, ax = fig_init()
# voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=1, line_alpha=0.5,
#                 point_size=0)
# plt.scatter(points[:, 0], points[:, 1], s=30, color='blue')
# plt.scatter(vertices[:, 0], vertices[:, 1], s=30, color='green')
# plt.xlim([-0.1, 1.1])
# plt.ylim([-0.1, 1.1])