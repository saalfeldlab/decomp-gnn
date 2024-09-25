import torch

from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.generators import data_generate
from ParticleGraph.models import data_train, data_test
from ParticleGraph.plotting import get_figures, load_and_display
from ParticleGraph.utils import set_device

config_list = ['arbitrary_3']
f_list = ['3']

for config_file in config_list:
    config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
    device = set_device(config.training.device)
    # data_generate(config, device=device, visualize=True, run_vizualized=0, style='color', alpha=1, erase=True, bSave=True, step=10) # config.simulation.n_frames // config.simulation.n_frames)
    # data_train(config, config_file, True, device)
    # data_test (config=config, config_file=config_file, visualize=True, style='color', verbose=False, best_model='0_7500', run=0, step=1, save_velocity=True, device=device) #config.simulation.n_frames // 3, test_simulation=False, sample_embedding=False, device=device)    # config.simulation.n_frames // 7

for f in f_list:
    config_list,epoch_list = get_figures(f, device=device)

load_and_display('graphs_data/graphs_arbitrary_3/Fig/Fig_0_0.tif', title="Initial configuration for data generation")
load_and_display('graphs_data/graphs_arbitrary_3/Fig/Fig_0_250.tif', title="Final configuration in data generation")
load_and_display('log/try_arbitrary_3/results/embedding_arbitrary_3_20.tif', title="Learned embedding")
load_and_display('log/try_arbitrary_3/results/func_all_arbitrary_3_20.tif', title="Learned functions")
load_and_display('log/try_arbitrary_3/tmp_recons/Fig_arbitrary_3_0.tif', title="Initial configuration for rollout")
load_and_display('log/try_arbitrary_3/tmp_recons/Fig_arbitrary_3_192.tif', title="Final configuration in rollout")
