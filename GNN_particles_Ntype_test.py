from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.generators.graph_data_generator import data_generate
from ParticleGraph.models import data_train, data_test
from ParticleGraph.models.Siren_Network import *
from ParticleGraph.utils import *


if __name__ == '__main__':

    try:
        matplotlib.use("Qt5Agg")
    except:
        pass

    config_list = ['arbitrary_3', 'boids_16_256']

    for config_file in config_list:
        # Load parameters from config file
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        # print(config.pretty())

        device = set_device(config.training.device)
        print(f'device {device}')

        # data_generate(config, device=device, visualize=True, run_vizualized=0, style='frame color', alpha=1, erase=True, bSave=True, step=8) #config.simulation.n_frames // 1)
        # data_train(config=config, config_file=config_file, erase=False, device=device)
        # data_test(config=config, config_file=config_file, visualize=True, style='latex frame color', verbose=False, best_model=20, run=1, step=config.simulation.n_frames // 25, test_simulation=False, sample_embedding=False, device=device)    # config.simulation.n_frames // 7



