from gflownet.proxy.base import Proxy
import numpy as np
import torch
from torchtyping import TensorType
import matplotlib.pyplot as plt
from botorch.test_functions.multi_fidelity import AugmentedHartmann
from mpl_toolkits.axes_grid1 import make_axes_locatable
from botorch.test_functions.synthetic import Hartmann as BotorchHartmann


class Hartmann(Proxy):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # This is just a rough estimate of modes
        self.modes = [
            [0.2, 0.2, 0.5, 0.3, 0.3, 0.7],
            [0.4, 0.9, 0.9, 0.6, 0.1, 0.0],
            [0.3, 0.1, 0.4, 0.3, 0.3, 0.7],
            [0.4, 0.9, 0.4, 0.6, 0.0, 0.0],
            [0.4, 0.9, 0.6, 0.6, 0.3, 0.0],
        ]

    # def plot_true_rewards(self, env, ax, rescale):
    #     states = torch.FloatTensor(env.get_all_terminating_states()).to(self.device)
    #     states_input_oracle = states.clone()
    #     states_oracle = env.statetorch2oracle(states_input_oracle)
    #     scores = self(states_oracle).detach().cpu().numpy()
    #     if hasattr(self, "fid"):
    #         title = "Oracle Energy (TrainY) with fid {}".format(self.fid)
    #     else:
    #         title = "Oracle Energy (TrainY)"
    #     # what the GP is trained on
    #     if self.maximize == False:
    #         scores = scores * (-1)
    #     index = states.long().detach().cpu().numpy()
    #     grid_scores = np.zeros((env.length, env.length))
    #     grid_scores[index[:, 0], index[:, 1]] = scores
    #     ax.set_xticks(
    #         np.arange(start=0, stop=env.length, step=int(env.length / rescale))
    #     )
    #     ax.set_yticks(
    #         np.arange(start=0, stop=env.length, step=int(env.length / rescale))
    #     )
    #     ax.imshow(grid_scores)
    #     ax.set_title(title)
    #     im = ax.imshow(grid_scores)
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes("right", size="5%", pad=0.05)
    #     plt.colorbar(im, cax=cax)
    #     plt.show()
    #     return ax


class MultiFidelityHartmann(Hartmann):
    def __init__(self, fid=None, cost=None, **kwargs):
        super().__init__(**kwargs)
        # maximization problem so negate = True
        self.task = AugmentedHartmann(negate=True)
        self.fid = fid
        self.cost = cost

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        fidelity = (
            torch.ones((len(states), 1), device=self.device, dtype=self.float)
            * self.fid
        )
        if isinstance(states, TensorType) == False:
            states = torch.tensor(states, device=self.device, dtype=self.float)
        else:
            states = states.to(self.device).to(self.float)
        if states.shape[1] != 2:
            states = states[:, :2]
        state_fid = torch.cat([states, fidelity], dim=1)
        scores = self.task(state_fid)
        # scores = scores.unsqueeze(-1)
        return scores.to(self.device).to(self.float)


class SingleFidelityHartmann(Hartmann):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # maximization problem so negate = True
        self.task = BotorchHartmann(negate=True)

    def __call__(self, states: TensorType["batch", "state_dim"]) -> TensorType["batch"]:
        if isinstance(states, TensorType) == False:
            states = torch.tensor(states, device=self.device, dtype=self.float)
        else:
            states = states.to(self.device).to(self.float)
        scores = self.task(states)
        return scores.to(self.device).to(self.float)
