import torch
import gpytorch
from tqdm import tqdm
import hydra
from botorch.models.gp_regression_fidelity import (
    SingleTaskMultiFidelityGP,
    SingleTaskGP,
)
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from abc import abstractmethod

"""
Assumes that in single fidelity, fid =1
"""


class SingleTaskGPRegressor:
    def __init__(self, logger, device, dataset, **kwargs):

        self.logger = logger
        self.device = device

        # Dataset
        self.dataset = dataset
        self.n_fid = dataset.n_fid
        self.n_samples = dataset.n_samples

        # Logger
        self.progress = self.logger.progress
        self.target_factor = -1

    @abstractmethod
    def init_model(self, train_x, train_y):
        # m is output dimension
        # TODO: if standardize is the desired operation
        pass

    def fit(self):
        train = self.dataset.train_dataset
        train_x = train["states"]
        train_y = train["energies"].unsqueeze(-1)
        # HACK: we want to maximise the energy, so we multiply by -1
        train_y = train_y * self.target_factor

        self.init_model(train_x, train_y)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        mll.to(train_x)
        mll = fit_gpytorch_mll(mll)

    def get_predictions(self, env, states):
        states_proxy_input = states.clone()
        states_proxy = env.statetorch2proxy(states_proxy_input)
        model = self.model
        model.eval()
        model.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = model.posterior(states_proxy)
            y_mean = posterior.mean
            y_std = posterior.variance.sqrt()
        y_mean = y_mean.detach().cpu().numpy().squeeze(-1)
        y_std = y_std.detach().cpu().numpy().squeeze(-1)
        return y_mean, y_std

    def get_metrics(self, y_mean, y_std, env, states):
        state_oracle_input = states.clone()
        if hasattr(env, "call_oracle_per_fidelity"):
            samples, fidelity = env.statebatch2oracle(state_oracle_input)
            targets = env.call_oracle_per_fidelity(samples, fidelity).detach().cpu()
        elif hasattr(env, "oracle"):
            samples = env.statebatch2oracle(state_oracle_input)
            targets = env.oracle(samples).detach().cpu()
        targets = targets * self.target_factor
        targets_numpy = targets.detach().cpu().numpy()
        targets_numpy = targets_numpy
        rmse = np.sqrt(np.mean((y_mean - targets_numpy) ** 2))
        nll = (
            -torch.distributions.Normal(torch.tensor(y_mean), torch.tensor(y_std))
            .log_prob(targets)
            .mean()
        )
        return rmse, nll

    def plot_predictions(self, states, scores, length, rescale=1):
        n_fid = self.n_fid
        if n_fid == 1:
            title = "GP Predictions"
        else:
            title = "GP Predictions with fid {}/{}".format(0, n_fid)
        n_states = int(length * length)
        if states.shape[-1] == 3:
            states = states[:, :2]
            states = torch.unique(states, dim=0)
        # states = states[:n_states]
        width = (n_fid) * 5
        fig, axs = plt.subplots(1, n_fid, figsize=(width, 5))
        for fid in range(0, n_fid):
            index = states.long().detach().cpu().numpy()
            grid_scores = np.zeros((length, length))
            grid_scores[index[:, 0], index[:, 1]] = scores[
                fid * n_states : (fid + 1) * n_states
            ]
            if n_fid == 1:
                ax = axs
            else:
                ax = axs[fid]
            ax.set_xticks(np.arange(start=0, stop=length, step=int(length / rescale)))
            ax.set_yticks(np.arange(start=0, stop=length, step=int(length / rescale)))
            ax.imshow(grid_scores)
            ax.set_title(title)
            im = ax.imshow(grid_scores)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()
        plt.tight_layout()
        plt.close()
        return fig

    def evaluate_model(self, env, rescale):
        states = torch.FloatTensor(env.get_all_terminating_states()).to("cuda")
        y_mean, y_std = self.get_predictions(env, states)
        rmse, nll = self.get_metrics(y_mean, y_std, env, states)
        figure = self.plot_predictions(states, y_mean, env.length, rescale)
        return figure, rmse, nll


class MultiFidelitySingleTaskRegressor(SingleTaskGPRegressor):
    def __init__(self, logger, device, dataset, **kwargs):
        super().__init__(logger, device, dataset, **kwargs)

    def init_model(self, train_x, train_y):
        self.model = SingleTaskMultiFidelityGP(
            train_x,
            train_y,
            outcome_transform=Standardize(m=1),
            # fid column
            data_fidelity=self.n_fid - 1,
        )


class SingleFidelitySingleTaskRegressor(SingleTaskGPRegressor):
    def __init__(self, logger, device, dataset, **kwargs):
        super().__init__(logger, device, dataset, **kwargs)

    def init_model(self, train_x, train_y):
        self.model = SingleTaskGP(
            train_x,
            train_y,
            outcome_transform=Standardize(m=1),
            # fid column
        )
