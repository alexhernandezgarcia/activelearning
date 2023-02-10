from gflownet.proxy.base import Proxy
import torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


class ToyOracle(Proxy):
    # TODO: resolve the kwargs error here
    def __init__(self, oracle, noise, env, valid, cost, device, float_precision):
        super().__init__(device, float_precision)
        self.oracle = oracle
        self.noise_distribution = torch.distributions.Normal(noise.mu, noise.sigma)
        self.sigma = noise.sigma
        self.valid = valid
        self.env = env
        self.cost = cost

    def __call__(self, states):
        true_values = self.oracle(states)
        if self.valid is not None:
            bounds = torch.FloatTensor(
                [[self.valid.xmin, self.valid.ymin], [self.valid.xmax, self.valid.ymax]]
            ).to(self.device)
            bounds = self.env.statetorch2oracle(bounds)
            mask = (states >= bounds[0]) & (states <= bounds[1])
            mask = mask[:, 0] & mask[:, 1]
            true_values[~mask] = 0
        noise = self.noise_distribution.sample(true_values.shape).to(self.device)
        noisy_values = true_values + noise
        return noisy_values

    def plot_scores(self):
        states = torch.FloatTensor(self.env.get_all_terminating_states()).to("cuda")
        scores = self(states)
        index = states.long().detach().cpu().numpy()
        grid_scores = np.ones((self.env.ndim, self.env.ndim)) * (0.2)
        grid_scores[index[:, 0], index[:, 1]] = scores
        plt.imshow(grid_scores)
        plt.colorbar()
        # TODO: fix to save to log directory/wandb logger
        plt.title("Ground Truth (with Noise Stddev {})".format(self.sigma))
        plt.savefig(
            "/home/mila/n/nikita.saxena/activelearning/storage/grid/round2/ground_truth_noise{}.png".format(
                self.sigma
            )
        )
        plt.close()
