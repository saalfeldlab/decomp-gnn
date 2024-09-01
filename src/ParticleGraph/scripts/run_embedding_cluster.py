# TODO: IN PROGRESS
from ParticleGraph.sparsify import *
from ParticleGraph.utils import to_numpy
from tqdm import trange

if __name__ == '__main__':

    # model_config = {'ninteractions': 3, 'nparticles': 4800, 'nparticle_types': 3, 'cmap': 'tab10', 'model':'PDE_A'}
    # model_config = {'ninteractions': 16, 'nparticles': 960, 'nparticle_types': 16, 'cmap': 'tab20', 'model':'GravityParticles'}
    model_config = {'ninteractions': 5, 'nparticles': 10000, 'nparticle_types': 5, 'cmap': 'tab20',
                    'model': 'RD_RPS_Mesh'}

    # cmap = cc(model_config=model_config)

    index_particles = []
    np_i = int(model_config['nparticles'] / model_config['nparticle_types'])
    for n in range(model_config['nparticle_types']):
        index_particles.append(np.arange(np_i * n, np_i * (n + 1)))

    embedding_cluster = EmbeddingCluster(model_config)
    nparticle_types = model_config['nparticle_types']
    nparticles = model_config['nparticles']

    # N = 100
    # data = np.random.randn(3 * N, 2)
    # data[:N] += 5
    # data[-N:] += 10
    # data[-1:] -= 20
    #
    # # clustering
    # thresh = 1.5
    # clusters, nclusters = embedding_cluster.get(data, method="distance")
    #
    # # plotting
    # plt.scatter(*np.transpose(data), c=clusters,s=5)
    # plt.axis("equal")
    # title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
    # plt.title(title)
    # plt.show()

    for epoch in range(20,21):

        # proj_interaction = np.load(f'/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/log/try_gravity_16c/tmp_training/umap_projection_{epoch}.npy')
        # proj_interaction = np.load(
        #     f'/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/log/try_arbitrary_3c/tmp_training/umap_projection_{epoch}.npy')
        proj_interaction = np.load(f'/home/allierc@hhmi.org/Desktop/Py/ParticleGraph/log/try_RD_RPS2b/tmp_training/umap_projection_{epoch}.npy')

        fig = plt.figure(figsize=(10, 3))
        ax = fig.add_subplot(1, 4, 1)
        plt.text(0, 1.1, f'Epoch: {epoch}', ha='left', va='top', transform=ax.transAxes, fontsize=10)
        plt.ion()
        # for thresh in trange(1,20):
        #     labels, nclusters = embedding_cluster.get(proj_interaction,'distance',thresh=thresh)
        #     plt.scatter(thresh, nclusters, s=10,c='k')

        ax = fig.add_subplot(1, 4, 2)
        for n in range(nparticle_types):
            plt.scatter(proj_interaction[index_particles[n], 0], proj_interaction[index_particles[n], 1], s=1)


        ax = fig.add_subplot(1, 4, 2)
        labels, nclusters = embedding_cluster.get(proj_interaction, 'distance', thresh=5)
        label_list = []
        for n in range(nclusters):
            pos = np.argwhere(labels == n)
            plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], s=1)
            print(pos.shape)
        for n in range(nparticle_types):
            tmp = labels[index_particles[n]]
            label_list.append(np.round(np.median(tmp)))
        label_list = np.array(label_list)
        new_labels = labels.copy()
        ax = fig.add_subplot(1, 4, 3)
        for n in range(nparticle_types):
            new_labels[labels == label_list[n]] = n
            pos = np.argwhere(labels == label_list[n])
            plt.scatter(proj_interaction[pos, 0], proj_interaction[pos, 1], s=0.1)
        plt.xlabel(r'UMAP 0', fontsize=12)
        plt.ylabel(r'UMAP 1', fontsize=12)
        plt.text(0., 1.1, f'Nclusters: {nclusters}', ha='left', va='top', transform=ax.transAxes)

        ax = fig.add_subplot(1, 4, 3)
        T1 = torch.zeros(int(nparticles / nparticle_types))
        for n in range(1, nparticle_types):
            T1 = torch.cat((T1, n * torch.ones(int(nparticles / nparticle_types))), 0)
        T1 = T1[:, None]
        confusion_matrix = metrics.confusion_matrix(to_numpy(T1), new_labels)  # , normalize='true')
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
        Accuracy = metrics.accuracy_score(to_numpy(T1), new_labels)
        plt.text(0, 1.1, f'Accuracy: {np.round(Accuracy,3)}', ha='left', va='top', transform=ax.transAxes, fontsize=10)
        cm_display.plot(ax=fig.gca(), cmap='Blues', include_values=False, values_format='d', colorbar=False)

        ax = fig.add_subplot(1, 4, 4)
        t = np.zeros((nparticle_types, 2))
        for n in range(nparticle_types):
            pos = np.argwhere(new_labels == n).squeeze().astype(int)
            temp = proj_interaction[pos, :]
            print(np.median(temp, axis=0))
            t[n,:] = np.median(temp, axis=0)
            plt.scatter(t[n, 0], t[n, 1], s=10)
        plt.tight_layout()

        plt.show()




