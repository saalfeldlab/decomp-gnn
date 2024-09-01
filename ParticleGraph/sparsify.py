import matplotlib.pyplot as plt
import numpy as np
import torch
import scipy.cluster.hierarchy as hcluster
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
from ParticleGraph.utils import *
import time


class EmbeddingCluster:
    def __init__(self, config):
        self.n_interactions = config.simulation.n_interactions
        self.cluster_connectivity = config.training.cluster_connectivity    # 'single' (default) or 'average'

    def get(self, data, method, thresh=2.5):

        match method:
            case 'kmeans':
                kmeans = KMeans(init="random", n_clusters=self.n_interactions, n_init=1000, max_iter=10000, random_state=10)
                k = kmeans.fit(data)
                clusters = k.labels_
                n_clusters = self.n_interactions
            case 'kmeans_auto':
                silhouette_avg_list = []
                silhouette_max = 0
                n_clusters = None
                for n in range(2, 10):
                    clusterer = KMeans(n_clusters=n, random_state=10, n_init='auto')
                    cluster_labels = clusterer.fit_predict(data)
                    silhouette_avg = silhouette_score(data, cluster_labels)
                    silhouette_avg_list.append(silhouette_avg)
                    if silhouette_avg > silhouette_max:
                        silhouette_max = silhouette_avg
                        n_clusters = n
                kmeans = KMeans(n_clusters=n_clusters, random_state=10, n_init='auto')
                k = kmeans.fit(data)
                clusters = k.labels_
            case 'distance':
                clusters = hcluster.fclusterdata(data, thresh, criterion="distance", method=self.cluster_connectivity) - 1
                n_clusters = len(np.unique(clusters))
            case 'inconsistent':
                clusters = hcluster.fclusterdata(data, thresh, criterion="inconsistent", method=self.cluster_connectivity) - 1
                n_clusters = len(np.unique(clusters))

            case _:
                raise ValueError(f'Unknown method {method}')

        return clusters, n_clusters


def sparsify_cluster(cluster_method, proj_interaction, embedding, cluster_distance_threshold, type_list, n_particle_types, embedding_cluster):

    # normalization of projection because UMAP output is not normalized
    proj_interaction = (proj_interaction - np.min(proj_interaction)) / (np.max(proj_interaction) - np.min(proj_interaction)+1e-10)
    embedding = (embedding - np.min(embedding)) / (np.max(embedding) - np.min(embedding)+1e-10)

    start_time = time.time()
    match cluster_method:
        case 'kmeans_auto_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'kmeans_auto')
        case 'kmeans_auto_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'kmeans_auto')
            proj_interaction = embedding
        case 'distance_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'distance', thresh=cluster_distance_threshold)
        case 'distance_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'distance', thresh=cluster_distance_threshold)
            proj_interaction = embedding
        case 'inconsistent_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'inconsistent', thresh=cluster_distance_threshold)
        case 'inconsistent_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'inconsistent', thresh=cluster_distance_threshold)
            proj_interaction = embedding
        case 'distance_both':
            new_projection = np.concatenate((proj_interaction, embedding), axis=-1)
            labels, n_clusters = embedding_cluster.get(new_projection, 'distance', thresh=cluster_distance_threshold)
    label_list = []
    for n in range(n_particle_types):
        pos = torch.argwhere(type_list == n)
        pos = to_numpy(pos)
        if len(pos) > 0:
            tmp = labels[pos[:,0]]
            label_list.append(np.round(np.median(tmp)))
            pos_ = np.argwhere(labels == np.median(tmp))

    label_list = np.array(label_list)

    fig,ax = fig_init()
    for n in label_list:
        pos = np.argwhere(labels == n)
        print(len(pos))
        if len(pos) > 0:
            ax.scatter(embedding[pos, 0], embedding[pos, 1], s=5)

    new_labels = np.ones_like(labels) * n_particle_types
    for n in range(n_particle_types):
        new_labels[labels == label_list[n]] = n

    computation_time = time.time() - start_time
    print(f"Clustering computation time is {computation_time} seconds.")


    fig,ax = fig_init()
    ax.scatter(proj_interaction[:, 0], proj_interaction[:, 1], c=new_labels, s=5, cmap='tab20')
    # plt.close()

    return labels, n_clusters, new_labels

def sparsify_cluster_state(cluster_method, proj_interaction, embedding, cluster_distance_threshold, true_type_list, n_particle_types, embedding_cluster):

    # normalization of projection because UMAP output is not normalized
    proj_interaction = (proj_interaction - np.min(proj_interaction)) / (np.max(proj_interaction) - np.min(proj_interaction)+1e-10)
    embedding = (embedding - np.min(embedding)) / (np.max(embedding) - np.min(embedding)+1e-10)

    start_time = time.time()
    match cluster_method:
        case 'kmeans_auto_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'kmeans_auto')
        case 'kmeans_auto_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'kmeans_auto')
            proj_interaction = embedding
        case 'distance_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'distance', thresh=cluster_distance_threshold)
        case 'distance_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'distance', thresh=cluster_distance_threshold)
            proj_interaction = embedding
        case 'inconsistent_plot':
            labels, n_clusters = embedding_cluster.get(proj_interaction, 'inconsistent', thresh=cluster_distance_threshold)
        case 'inconsistent_embedding':
            labels, n_clusters = embedding_cluster.get(embedding, 'inconsistent', thresh=cluster_distance_threshold)
            proj_interaction = embedding
        case 'distance_both':
            new_projection = np.concatenate((proj_interaction, embedding), axis=-1)
            labels, n_clusters = embedding_cluster.get(new_projection, 'distance', thresh=cluster_distance_threshold)

    computation_time = time.time() - start_time
    print(f"Clustering computation time is {computation_time} seconds.")

    label_list = []
    for n in range(n_particle_types):
        pos = np.argwhere(true_type_list == n).squeeze().astype(int)
        if len(pos)>0:
            tmp = labels[pos]
            label_list.append(np.round(np.median(tmp)))
        else:
            label_list.append(0)
    label_list = np.array(label_list)
    new_labels = np.ones_like(labels) * n_particle_types
    for n in range(n_particle_types):
        new_labels[labels == label_list[n]] = n

    return labels, n_clusters, new_labels



if __name__ == '__main__':
    # generate 3 clusters of each around 100 points and one orphan point
    n_interactions = 3
    embedding_cluster = EmbeddingCluster(n_interactions)

    N = 100
    data = np.random.randn(3 * N, 2)
    data[:N] += 5
    data[-N:] += 10
    data[-1:] -= 20

    # clustering
    thresh = 1.5
    clusters, n_clusters = embedding_cluster.get(data, method="distance")

    # plotting
    plt.scatter(*np.transpose(data), c=clusters, s=5)
    plt.axis("equal")
    title = "threshold: %f, number of clusters: %d" % (thresh, n_clusters)
    plt.title(title)
    plt.show()
