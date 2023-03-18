from gflownet.envs.grid import Grid as GflowNetGrid
import torch
from torchtyping import TensorType
from typing import List, Tuple
from .base import GFlowNetEnv


class Grid(GFlowNetEnv, GflowNetGrid):
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs,
        )
        if self.proxy_state_format == "ohe":
            self.statebatch2proxy = self.statebatch2policy
        elif self.proxy_state_format == "oracle":
            self.statebatch2proxy = self.statebatch2oracle
            self.statetorch2proxy = self.statetorch2oracle
        elif "state" in self.proxy_state_format:
            # As We want it to be compatible with both state_fidId and state formats
            # state is for when proxy is GP so fidelities are fractional
            # state_fidIdx is for the multifidelity GP where an index kernel is used
            self.statebatch2proxy = self.statebatch2state
            self.statetorch2proxy = self.statetorch2state
            # Assumes that the oracle is always Branin
            self.statebatch2oracle = self.statebatch2state
            self.statetorch2oracle = self.statetorch2state
        else:
            raise NotImplementedError(
                f"Proxy state format {self.proxy_state_format} not implemented"
            )

    def statebatch2state(self, state_batch):
        """
        Converts a batch of states to AugmentedBranin oracle format
        """
        if isinstance(state_batch, torch.Tensor) == False:
            state_batch = torch.tensor(state_batch)
        return self.statetorch2state(state_batch)

    def statetorch2state(self, state_torch):
        """
        Converts a batch of states to AugmentedBranin oracle format
        """
        state_torch = state_torch / self.rescale
        return state_torch.to(self.float).to(self.device)
