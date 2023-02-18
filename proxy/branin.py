from gflownet.proxy.base import Proxy
import numpy as np
import torch
from torchtyping import TensorType
import matplotlib.pyplot as plt
from botorch.test_functions.multi_fidelity import AugmentedBranin
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Branin(Proxy):
    def __init__(self, fid=None, cost=None, env=None, **kwargs):
        super().__init__(**kwargs)
        # minimisation problem so negate = False
        self.task = AugmentedBranin(negate=False)
        self.fid = fid
        self.cost = cost

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        fidelity = (
            torch.ones((len(states), 1), device=self.device, dtype=self.float)
            * self.fid
        )
        if isinstance(states, TensorType) == False:
            states = torch.tensor(states, device=self.device, dtype=self.float)
        # fidelity[:,0] = fidelity[:, 0] - 0.1
        # states = states - 5
        state_fid = torch.cat([states, fidelity], dim=1)
        scores = self.task(state_fid)
        # scores = scores.unsqueeze(-1)
        return scores.to(self.device).to(self.float)

    def plot_true_rewards(self, env, ax, rescale):
        states = torch.FloatTensor(env.get_all_terminating_states()).to(self.device)
        states = states / rescale
        scores = self(states).detach().cpu().numpy()
        # what the GP is trained on
        scores = scores * (-0.1)
        index = states.long().detach().cpu().numpy()
        grid_scores = np.zeros((env.length, env.length))
        grid_scores[index[:, 0], index[:, 1]] = scores
        ax.set_xticks(np.arange(env.length))
        ax.set_yticks(np.arange(env.length))
        ax.imshow(grid_scores)
        ax.set_title("True Reward with fid {}".format(self.fid))
        im = ax.imshow(grid_scores)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()
        return ax
