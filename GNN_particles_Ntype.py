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

    config_list = ['arbitrary_3']

    for config_file in config_list:
        config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
        device = set_device(config.training.device)
        print(f'device {device}')
        # data_generate(config, device=device, visualize=True, run_vizualized=0, style='color', alpha=1, erase=True, bSave=True, step=10) # config.simulation.n_frames // config.simulation.n_frames)
        # data_train(config, config_file, False, device)
        # data_test (config=config, config_file=config_file, visualize=True, style='color', verbose=False, best_model='0_7500', run=0, step=1, save_velocity=True, device=device) #config.simulation.n_frames // 3, test_simulation=False, sample_embedding=False, device=device)    # config.simulation.n_frames // 7
