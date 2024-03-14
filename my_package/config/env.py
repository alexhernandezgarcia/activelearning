from dataclasses import dataclass
from typing import Optional



@dataclass
class BufferSpec:
    path: Optional[str] = None
    type: Optional[str] = None
    output_pkl: Optional[str] = None
    n: Optional[int] = None
    seed: Optional[int] = None
    output_csv: Optional[str] = None 
   

@dataclass
class Buffer:
    train: BufferSpec
    test: BufferSpec
    replay_capacity: float = 10
   

@dataclass
class Environment:
    # Buffer
    buffer: Buffer
    min_reward: float = 1e-15
    corr_type: Optional[str] = None
    length: int = 100
    rescale: int = 10
    # Reward function: power or boltzmann
    # boltzmann: exp(-1.0 * reward_beta * proxy)
    # power: (-1.0 * proxy / reward_norm) ** self.reward_beta
    reward_func: str = "power"
    # Beta parameter of the reward function
    reward_beta: float = 1.0
    # Reward normalization for "power" reward function
    reward_norm: float = 1.0
    # If > 0, reward_norm = reward_norm_std_mult * std(energies)
    reward_norm_std_mult: int = 8
    # Supported options: state, oracle or ohe
    proxy_state_format: str = "state"


@dataclass
class Grid_Env(Environment):
    id: str = "grid"
    func: str = "corners"
    # Dimensions of hypergrid
    n_dim: int = 2
    # Number of cells per dimension
    length: int = 20
    # Minimum and maximum number of steps in the action space
    min_step_len: int = 1
    max_step_len: int = 1
    # Mapping coordinates
    cell_min: int = -1
    cell_max: int = 1