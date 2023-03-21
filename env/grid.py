from gflownet.envs.grid import Grid as GflowNetGrid
import torch
from torchtyping import TensorType
from typing import List, Tuple
from .base import GFlowNetEnv
import matplotlib.pyplot as plt


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

    def plot_reward_distribution(
        self, states=None, scores=None, ax=None, title=None, oracle=None, **kwargs
    ):
        if ax is None:
            fig, ax = plt.subplots()
            standalone = True
        else:
            standalone = False
        if title == None:
            title = "Scores of Sampled States"
        if oracle is None:
            oracle = self.oracle
        if scores is None:
            oracle_states = self.statetorch2oracle(states)
            scores = oracle(oracle_states)
        if isinstance(scores, TensorType):
            scores = scores.cpu().detach().numpy()
        ax.hist(scores)
        ax.set_title(title)
        ax.set_ylabel("Number of Samples")
        ax.set_xlabel("Energy")
        plt.show()
        if standalone == True:
            plt.tight_layout()
            plt.close()
        return ax

    def plot_samples_frequency(self, samples, ax=None, title=None, rescale=1):
        """
        Plot 2D histogram of samples.
        """
        if self.n_dim > 2:
            return None
        if ax is None:
            fig, ax = plt.subplots()
            standalone = True
        else:
            standalone = False
        # assuming the first time this function would be called when the dataset is created
        if self.rescale == None:
            self.rescale = rescale
        # make a list of integers from 0 to n_dim
        if self.rescale != 1:
            step = int(self.length / self.rescale)
        else:
            step = 1
        ax.set_xticks(np.arange(start=0, stop=self.length, step=step))
        ax.set_yticks(np.arange(start=0, stop=self.length, step=step))
        # check if samples is on GPU
        if torch.is_tensor(samples) and samples.is_cuda:
            samples = samples.detach().cpu()
        states = np.array(samples).astype(int)
        grid = np.zeros((self.length, self.length))
        if title == None:
            ax.set_title("Frequency of Coordinates Sampled")
        else:
            ax.set_title(title)
        # TODO: optimize
        for state in states:
            grid[state[0], state[1]] += 1
        im = ax.imshow(grid)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()
        if standalone == True:
            plt.tight_layout()
            plt.close()
        return ax