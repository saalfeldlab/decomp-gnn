# %% [markdown]
# ---
# title: Figure 3
# author: Cédric Allier, Michael Innerberger, Stephan Saalfeld
# execute:
#   echo: false
# ---

# %% [markdown]
# This script creates the first column of Figure 3 in the paper: we look at an attraction-repulsion system with three
# particle types.

# %%
#| output: false
import torch

from ParticleGraph.config import ParticleGraphConfig
from ParticleGraph.generators import data_generate
from ParticleGraph.models import data_train, data_test
from ParticleGraph.plotting import get_figures, load_and_display
from ParticleGraph.utils import set_device

# %%
#| echo: true
#| output: false
# First, we load the configuration file and set the device.
config_file = 'arbitrary_3'
figure_id = '3'
config = ParticleGraphConfig.from_yaml(f'./config/{config_file}.yaml')
device = set_device("auto")

# %%
#| echo: true
#| eval: false
# Subsequently, the data is generated, and the model is trained and tested.
# Since we ship the trained model with the repository, this step can be skipped if desired.
data_generate(config, device=device, visualize=True, run_vizualized=0, style='color', alpha=1, erase=True, bSave=True, step=10)
data_train(config, config_file, True, device)
data_test(config=config, config_file=config_file, visualize=True, style='color', verbose=False, best_model='0_7500', run=0, step=1, save_velocity=True, device=device)

# %%
#| echo: true
#| eval: false
# Here, we generate the figures that are shown in the first column of Figure 3. The model that has been trained in the
# previous step is used to generate the rollouts.
config_list, epoch_list = get_figures(figure_id, device=device)

# %%
#| fig-cap: "Initial configuration for data generation. The orange, blue, and green particles represent the three different particle types."
load_and_display('graphs_data/graphs_arbitrary_3/Fig/Fig_0_0.tif')

# %%
#| fig-cap: "Final configuration after data generation"
load_and_display('graphs_data/graphs_arbitrary_3/Fig/Fig_0_250.tif')

# %%
#| fig-cap: "Learned embedding of the particle types"
load_and_display('log/try_arbitrary_3/results/embedding_arbitrary_3_20.tif')

# %%
#| fig-cap: "Learned interaction functions"
load_and_display('log/try_arbitrary_3/results/func_all_arbitrary_3_20.tif')

# %%
#| fig-cap: "Initial random configuration for rollout"
load_and_display('log/try_arbitrary_3/tmp_recons/Fig_arbitrary_3_0.tif')

# %%
#| fig-cap: "Final configuration in rollout, which looks qualitatively very similar to the final configuration of the data generation"
load_and_display('log/try_arbitrary_3/tmp_recons/Fig_arbitrary_3_192.tif')
