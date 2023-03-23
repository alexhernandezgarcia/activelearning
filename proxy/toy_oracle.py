from gflownet.proxy.base import Proxy
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


class ToyOracle(Proxy):
    # TODO: resolve the kwargs error here
    def __init__(
        self, oracle, noise, env, valid, cost, fid, device, float_precision, maximize
    ):
        super().__init__(device, float_precision, maximize)
        self.oracle = oracle
        self.noise_distribution = torch.distributions.Normal(noise.mu, noise.sigma)
        self.sigma = noise.sigma
        self.valid = valid
        self.env = env
        self.cost = cost
        self.fid = fid

    def __call__(self, states):
        true_values = self.oracle(states)
        if self.valid is not None:
            bounds = torch.tensor(
                [
                    [self.valid.xmin, self.valid.ymin],
                    [self.valid.xmax, self.valid.ymax],
                ],
                device=self.device,
                dtype=self.float,
            )
            bounds = self.env.statetorch2oracle(bounds)
            mask = (states >= bounds[0]) & (states <= bounds[1])
            mask = mask[:, 0] & mask[:, 1]
            true_values[~mask] = 0
        noise = self.noise_distribution.sample(true_values.shape).to(self.device)
        noisy_values = true_values + noise
        return noisy_values

    def plot_true_rewards(self, env, ax, **kwargs):
        states = torch.tensor(
            env.get_all_terminating_states(), dtype=self.float, device=self.device
        )
        states_oracle = env.statetorch2oracle(states)
        scores = self(states_oracle).detach().cpu().numpy()
        scores = scores * (-1)
        index = states.long().detach().cpu().numpy()
        grid_scores = np.zeros((env.length, env.length))
        grid_scores[index[:, 0], index[:, 1]] = scores
        ax.set_xticks(np.arange(env.length))
        ax.set_yticks(np.arange(env.length))
        ax.imshow(grid_scores)
        ax.set_title("Oracle Energy (TrainY) with fid {}".format(self.fid))
        im = ax.imshow(grid_scores)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()
        plt.close()
        return ax
