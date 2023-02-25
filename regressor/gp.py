import torch
import gpytorch
from tqdm import tqdm
import hydra
from botorch.models.gp_regression_fidelity import (
    SingleTaskMultiFidelityGP,
)
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
Assumes that in single fidelity, fid =1
"""


class MultitaskGPRegressor:
    def __init__(self, logger, device, dataset, **kwargs):

        self.logger = logger
        self.device = device

        # Dataset
        self.dataset = dataset
        self.n_fid = dataset.n_fid
        self.n_samples = dataset.n_samples

        # Logger
        self.progress = self.logger.progress

    def init_model(self, train_x, train_y):
        # m is output dimension
        # TODO: if standardize is the desired operation
        self.model = SingleTaskMultiFidelityGP(
            train_x,
            train_y,
            outcome_transform=Standardize(m=1),
            # fid column
            data_fidelity=self.n_fid - 1,
        )

    def fit(self):
        train = self.dataset.train_dataset
        train_x = train["samples"]
        # HACK: we want to maximise the energy, so we multiply by -1
        train_y = train["energies"].unsqueeze(-1)
        train_y = train_y * (-1)

        self.init_model(train_x, train_y)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        mll.to(train_x)
        mll = fit_gpytorch_mll(mll)

    def get_predictions(self, states):
        model = self.model
        model.eval()
        model.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = model.posterior(states)
            y_mean = posterior.mean
            y_std = posterior.variance.sqrt()
        y_mean = y_mean.detach().cpu().numpy().squeeze(-1)
        y_std = y_std.detach().cpu().numpy().squeeze(-1)
        return y_mean, y_std

    def get_metrics(self, env, states):
        y_mean, y_std = self.get_predictions(states)
        if hasattr(env, "call_oracle_per_fidelity"):
            states, fidelity = env.statebatch2oracle(states)
            targets = env.call_oracle_per_fidelity(states, fidelity).detach().cpu()
        elif hasattr(env, "oracle"):
            targets = env.oracle(states).detach().cpu()
        targets_numpy = targets.detach().cpu().numpy()
        # compute rmse between two numpy arrays
        rmse = np.sqrt(np.mean((y_mean - targets_numpy) ** 2))
        nll = (
            -torch.distributions.Normal(torch.tensor(y_mean), torch.tensor(y_std))
            .log_prob(targets)
            .mean()
        )
        return rmse, nll

    def plot_predictions(self, states, length, rescale=None):
        n_fid = self.n_fid
        scores, _ = self.get_predictions(states)
        n_states = int(length * length)
        states = states[:n_states]
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
            ax.set_title("GP Predictions with fid {}/{}".format(fid, n_fid))
            im = ax.imshow(grid_scores)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()
        plt.tight_layout()
        plt.close()
        return fig

    def evaluate_model(self, env, rescale):
        n_fid = self.n_fid
        # if n_fid == 1:
        #     sfenv = env
        # else:
        #     mfenv = env
        #     env = env.env
        # if hasattr(env, "get_all_terminating_states") == False:
        #     return None
        states = torch.FloatTensor(env.get_all_terminating_states()).to("cuda")
        # n_states = states.shape[0]
        if n_fid > 1:
            # fidelities = torch.zeros((len(states) * n_fid, 1)).to("cuda")
            # for i in range(n_fid):
            #     fidelities[i * len(states) : (i + 1) * len(states), 0] = mfenv.oracle[
            #         i
            #     ].fid
            # states = states.repeat(n_fid, 1)
            # state_fid = torch.cat([states, fidelities], dim=1)
            state_fid = states
            state_fid_proxy = env.statetorch2proxy(state_fid)
            # if isinstance(state_fid_proxy, tuple):
            #     state_fid_proxy = torch.concat(
            #         (state_fid_proxy[0], state_fid_proxy[1]), dim=1
            #     )
        else:
            state_fid = states
            state_fid_proxy = env.statetorch2proxy(state_fid)
            # fidelities = torch.ones((len(states), 1)).to(states.device) * env.oracle.fid
            # state_fid_proxy = torch.cat([state_fid_proxy, fidelities], dim=1)
        figure = self.plot_predictions(state_fid_proxy, env.length, rescale)
        rmse, nll = self.get_metrics(env, state_fid_proxy)
        return figure, rmse, nll
