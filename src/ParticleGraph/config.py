from typing import Optional, Literal, Annotated, Dict

import yaml
from pydantic import BaseModel, ConfigDict, Field

# from ParticleGraph import (
#     GraphModel,
# )


# Sub-config schemas for ParticleGraph

class SimulationConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    dimension: int = 2
    connectivity_file: str = ''
    connectivity_init: list[float] =[-1]
    connectivity_mask : str=''
    excitation: str='none'
    adjacency_matrix: str = ''
    phi: str = 'tanh'
    tau: float = 1.0
    params: list[list[float]]
    cell_cycle_length: list[float] =[-1]
    cell_death_rate: list[float] = [-1]
    cell_area: list[float] = [-1]
    min_radius: Annotated[float, Field(ge=0)] = 0
    max_radius: Annotated[float, Field(gt=0)]
    n_neighbors: int = 10
    angular_sigma: float = 0
    angular_Bernouilli: list[float] =[-1]
    max_edges: float = 1.0E6
    diffusion_coefficients: list[list[float]] = None
    n_particles: int = 1000
    n_particles_max: int = 20000
    n_particle_types: int = 5
    n_interactions: int = 5
    has_cell_state: bool = False
    state_type: Literal['discrete', 'sequence', 'continuous'] = 'discrete'
    state_params: list[float] =[-1]
    non_discrete_level: float = 0
    n_nodes: Optional[int] = None
    n_node_types: Optional[int] = None
    pos_rate: list[list[float]] = None
    neg_rate: list[list[float]] = None
    has_cell_division: bool = False
    has_cell_death: bool = False
    cell_inert_model_coeff: float = 0
    coeff_area: float = 1
    coeff_perimeter: float = 0
    cell_active_model_coeff: float = 1
    n_frames: int = 1000
    sigma: float = 0.005
    delta_t: float = 1
    dpos_init: float = 0
    boundary: Literal['periodic', 'no', 'periodic_special'] = 'periodic'
    cell_type_map: Optional[str] = None
    node_coeff_map: Optional[str] = None
    node_value_map: Optional[str] = None
    node_proliferation_map: Optional[str] = None
    beta: Optional[float] = None
    start_frame: int = 0
    final_cell_mass: list[float] = [-1]
    mc_slope: list[float] = [-1]
    has_fluo: bool = False
    fluo_path: str = ''
    fluo_method: str = 'padding'

class GraphModelConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    particle_model_name: str = ''
    cell_model_name: str = ''
    mesh_model_name: str = ''
    signal_model_name: str = ''
    prediction: Literal['first_derivative', '2nd_derivative'] = '2nd_derivative'
    input_size: int
    output_size: int
    hidden_dim: int
    n_mp_layers: int
    aggr_type: str
    mesh_aggr_type: str = 'add'
    embedding_dim: int = 2

    update_type: Literal['linear', 'none', 'embedding_MLP'] = 'none'
    input_size_update: int = 3
    n_layers_update: int = 3
    hidden_dim_update: int = 64
    output_size_update: int = 1

    input_size_nnr: int = 3
    n_layers_nnr: int = 5
    hidden_dim_nnr: int = 128
    output_size_nnr: int = 1


    field_method: Literal['none', 'tensor', 'Siren_wo_time', 'Siren_with_time'] = 'tensor'

    division_predictor_input_size: int = 3
    division_predictor_hidden_dim: int = 64
    division_predictor_n_layers: int = 3
    division_predictor_output_size: int = 1

    field_type: str = 'tensor'


    # def get_instance(self, **kwargs):
    #     return GraphModel(**self.model_dump(), **kwargs)


class PlottingConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    colormap: str = 'tab10'
    arrow_length: int = 10
    marker_size: int = 100
    ylim: list[float] = [-0.1, 0.1]
    embedding_lim: list[float] = [-40, 40]


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    n_epochs: int = 20
    batch_size: int = 1
    small_init_batch_size: bool = True
    large_range: bool = False
    do_tracking: bool = False

    n_runs: int = 2
    seed : int = 40
    clamp: float = 0
    pred_limit: float = 1.E+10
    sparsity: Literal['none', 'replace_embedding', 'replace_embedding_function', 'replace_state', 'replace_track'] = 'none'
    use_hot_encoding: bool = False
    sparsity_freq : int = 5
    particle_dropout: float = 0
    n_ghosts: int = 0
    ghost_method: Literal['none', 'tensor', 'MLP'] = 'none'
    ghost_logvar: float = -12

    fix_cluster_embedding: bool = False
    loss_weight: bool = False
    learning_rate_start: float = 0.001
    learning_rate_end: float = 0.0005
    learning_rate_embedding_start: float = 0.001
    learning_rate_embedding_end: float = 0.001

    coeff_L1: float = 0
    coeff_loss1: float = 1
    coeff_loss2: float = 1
    coeff_loss3: float = 1

    noise_level: float = 0
    data_augmentation: bool = True
    data_augmentation_loop: int = 40
    recursive_loop: int = 0
    regul_matrix: bool = False
    sub_batches: int = 1

    sequence: list[str] = ['to track','to cell']

    cluster_method: Literal['kmeans', 'kmeans_auto_plot', 'kmeans_auto_embedding', 'distance_plot', 'distance_embedding', 'distance_both', 'inconsistent_plot', 'inconsistent_embedding', 'none'] = 'distance_plot'
    cluster_distance_threshold: float = 0.01
    cluster_connectivity: Literal['single','average'] = 'single'

    state_hot_encoding: bool = False
    state_temperature: float = 0.5

    device: Annotated[str, Field(pattern=r'^(auto|cpu|cuda:\d+)$')] = 'auto'


# Main config schema for ParticleGraph

class ParticleGraphConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    description: Optional[str] = 'ParticleGraph'
    dataset: str
    data_folder_name: str = 'none'
    simulation: SimulationConfig
    graph_model: GraphModelConfig
    plotting: PlottingConfig
    training: TrainingConfig

    @staticmethod
    def from_yaml(file_name: str):
        with open(file_name, 'r') as file:
            raw_config = yaml.safe_load(file)
        return ParticleGraphConfig(**raw_config)

    def pretty(self):
        return yaml.dump(self, default_flow_style=False, sort_keys=False, indent=4)


if __name__ == '__main__':

    config_file = '../../config/arbitrary_3.yaml' # Insert path to config file
    config = ParticleGraphConfig.from_yaml(config_file)
    print(config.pretty())

    print('Successfully loaded config file. Model description:', config.description)
    
