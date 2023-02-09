from gflownet.proxy.base import Proxy
import torch
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import gpytorch
from mpl_toolkits.axes_grid1 import make_axes_locatable


class ToyOracle(Proxy):
    # TODO: resolve the kwargs error here
    def __init__(self, oracle, config, env, device, float_precision):
        super().__init__(device, float_precision)
        self.oracle = oracle
        if config.noise.sigma != 0.0:
            self.noise_distribution = torch.distributions.Normal(
                config.noise.mu, config.noise.sigma
            )
        else:
            self.noise_distribution = None
        # self.mu =  noise.mu
        self.sigma = config.noise.sigma
        self.valid = config.valid
        self.env = env
        self.cost = config.cost
        # states = env.get_all_terminating_states()
        # noise = self.noise_distribution.sample(states.shape).to(self.device)
        # self.state_noise_dict = {
        # "state": states,
        # "noise": noise,
        # }

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
        if self.noise_distribution is not None:
            noise = self.noise_distribution.sample(true_values.shape).to(self.device)
            noisy_values = true_values + noise
            return noisy_values
        else:
            return true_values

    def plot_scores(self):
        states = torch.FloatTensor(self.env.get_all_terminating_states()).to("cuda")
        scores = self(states)
        index = states.long().detach().cpu().numpy()
        grid_scores = np.ones((self.env.ndim, self.env.ndim)) * (0.2)
        grid_scores[index[:, 0], index[:, 1]] = scores
        plt.imshow(grid_scores)
        plt.colorbar()
        plt.title("Ground Truth (with Noise Stddev {})".format(self.sigma))
        plt.savefig(
            "/home/mila/n/nikita.saxena/activelearning/storage/grid/round2/ground_truth_noise{}.png".format(
                self.sigma
            )
        )
        plt.close()


def make_dataset(env, oracles, n_fid, device, path):
    # get_states return a list right now
    # states = torch.Tensor(env.env.get_uniform_terminating_states(200)).to(device).long()
    states = (
        torch.Tensor(np.random.randint(low=0, high=10, size=(100, 2))).to(device).long()
    )
    fidelities = torch.randint(0, n_fid, (len(states), 1)).to(device)
    state_fid = torch.cat([states, fidelities], dim=1)
    states_fid_oracle = env.statetorch2oracle(state_fid)
    scores = env.call_oracle_per_fidelity(states_fid_oracle)
    states_fid_list = state_fid.detach().tolist()
    readable_states = [env.state2readable(state) for state in states_fid_list]
    df = pd.DataFrame(
        {"samples": readable_states, "scores": scores.detach().cpu().tolist()}
    )
    df.to_csv(path)


def plot_predictions(env, regressor, fid):
    """Plot the predictions of the regressor for a given fidelity."""
    states = torch.FloatTensor(env.env.get_all_terminating_states()).to("cuda")
    fidelities = torch.ones((len(states), 1)).to("cuda") * fid
    state_fid = torch.cat([states, fidelities], dim=1)
    state_fid_proxy = env.statetorch2proxy(state_fid)
    scores = regressor.model(state_fid_proxy).detach().cpu().numpy()

    grid_scores = np.reshape(scores, (20, 20))
    plt.imshow(grid_scores)
    plt.colorbar()
    plt.title("Predictions for fid {}".format(fid))
    # plt.show()
    # save the figure
    plt.savefig(
        "/home/mila/n/nikita.saxena/activelearning/storage/grid/dummy_test_{}.png".format(
            fid
        )
    )
    plt.close()


def plot_predictions_oracle(env, fid):
    """Plot the ground truth for a given fidelity."""
    states = torch.FloatTensor(env.env.get_all_terminating_states()).to("cuda")
    fidelities = torch.ones((len(states), 1)).to("cuda") * fid
    state_fid = torch.cat([states, fidelities], dim=1)
    states_fid_oracle = env.statetorch2oracle(state_fid)
    scores = env.call_oracle_per_fidelity(states_fid_oracle).detach().cpu().numpy()
    index = states.long().detach().cpu().numpy()
    # data_scores = np.reshape(scores.detach().cpu().numpy(), (10, 10))
    grid_scores = np.ones((20, 20)) * (0.2)
    grid_scores[index[:, 0], index[:, 1]] = scores
    plt.imshow(grid_scores)
    plt.colorbar()
    plt.title("Ground Truth for fid {}".format(fid))
    plt.savefig(
        "/home/mila/n/nikita.saxena/activelearning/storage/grid/gp/final_ground_truth_{}.png".format(
            fid
        )
    )
    plt.close()


def plot_acquisition(env, fid, mf_mes_oracle):
    states = torch.FloatTensor(env.env.get_all_terminating_states()).to("cuda")
    fidelities = torch.ones((len(states), 1)).to("cuda") * fid
    state_fid = torch.cat([states, fidelities], dim=1)
    states_fid_oracle = env.statetorch2oracle(state_fid)
    scores = mf_mes_oracle(states_fid_oracle).detach().cpu().numpy()
    index = states.long().detach().cpu().numpy()
    grid_scores = np.ones((20, 20)) * (0.2)
    grid_scores[index[:, 0], index[:, 1]] = scores
    plt.imshow(grid_scores)
    plt.colorbar()
    plt.title("Reward with fid {}".format(fid))
    plt.savefig(
        "/home/mila/n/nikita.saxena/activelearning/storage/grid/gp/updated_cost_fid{}.png".format(
            fid
        )
    )
    plt.close()


def plot_context_points(env, mf_mes_oracle):
    # states = torch.Tensor(env.env.get_uniform_terminating_states(100)).to("cuda").long()
    states = (
        torch.Tensor(np.random.randint(low=0, high=10, size=(100, 2))).to("cuda").long()
    )
    fidelities = torch.randint(0, 3, (len(states), 1)).to("cuda")
    state_fid = torch.cat([states, fidelities], dim=1)
    states_fid_oracle = env.statetorch2oracle(state_fid)
    context_points = mf_mes_oracle.candidate_set
    scores = mf_mes_oracle(context_points).detach().cpu().numpy()
    grid_scores = np.ones((20, 20)) * (-0.0001)
    index = states.long().detach().cpu().numpy()
    grid_scores[index[:, 0], index[:, 1]] = scores
    plt.imshow(grid_scores)
    plt.colorbar()
    plt.title("Context points Lower Quadrant")
    plt.savefig(
        "/home/mila/n/nikita.saxena/activelearning/storage/grid/gp/updated_cost_context.png"
    )
    plt.close()


def plot_gp_predictions(env, regressor):
    """Plot the predictions of the regressor for a given fidelity."""
    states = torch.FloatTensor(env.env.get_all_terminating_states()).to("cuda")
    n_states = states.shape[0]
    n_fid = 3
    model = regressor.model
    model.eval()
    model.likelihood.eval()
    fidelities = torch.zeros((len(states) * 3, 1)).to("cuda")
    for i in range(n_fid):
        fidelities[i * len(states) : (i + 1) * len(states), 0] = i
    states = states.repeat(n_fid, 1)
    state_fid = torch.cat([states, fidelities], dim=1)
    state_fid_proxy = env.statetorch2oracle(state_fid)
    states = states[:n_states]

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = model.likelihood(model(state_fid_proxy))
        mean = predictions.mean
    scores = mean.detach().cpu().numpy()
    width = (n_fid) * 5
    fig, axs = plt.subplots(1, n_fid, figsize=(width, 5))
    for fid in range(0, n_fid):
        index = states.long().detach().cpu().numpy()
        grid_scores = np.zeros((env.env.length, env.env.length))
        grid_scores[index[:, 0], index[:, 1]] = scores[
            fid * len(states) : (fid + 1) * len(states)
        ]
        axs[fid].set_xticks(np.arange(env.env.length))
        axs[fid].set_yticks(np.arange(env.env.length))
        axs[fid].imshow(grid_scores)
        axs[fid].set_title("GP Predictions with fid {}".format(fid))
        im = axs[fid].imshow(grid_scores)
        divider = make_axes_locatable(axs[fid])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        plt.show()
    plt.tight_layout()
    plt.close()
    return fig
    # fidelities = torch.ones((len(states), 1)).to("cuda") * fid
    # state_fid = torch.cat([states, fidelities], dim=1)
    # state_fid_proxy = env.statetorch2proxy(state_fid)
    # model = regressor.model
    # model.eval()
    # model.likelihood.eval()

    # # Make predictions
    # with torch.no_grad(), gpytorch.settings.fast_pred_var():
    #     predictions = model.likelihood(model(state_fid_proxy))
    #     mean = predictions.mean
    # mean = mean.detach().cpu().numpy()
    # index = states.long().detach().cpu().numpy()
    # grid_scores = np.zeros((env.env.length, env.env.length))
    # grid_scores[index[:, 0], index[:, 1]] = mean
    # plt.imshow(grid_scores)
    # plt.colorbar()
    # plt.title("GP Predictions for fid {}".format(fid))
    # plt.savefig(
    #     "/home/mila/n/nikita.saxena/activelearning/storage/grid/gp/no_norm_gp_prediction_{}.png".format(
    #         fid
    #     )
    # )
    # plt.close()

    # scores = mf_mes_oracle(states_fid_oracle).detach().cpu().numpy()
    # for fid in range(3):
    #     idx_fid = torch.where(fidelities == fid)[0]
    #     grid_scores = np.ones((20, 20)) * (-0.01)
    #     states = state_fid[idx_fid].long().detach().cpu().numpy()
    #     grid_scores[states[:, 0], states[:, 1]] = scores[idx_fid.detach().cpu().numpy()]
    #     plt.imshow(grid_scores)
    #     plt.colorbar()
    #     plt.title("Context points for fid {}".format(fid))
    #     plt.savefig(
    #         "/home/mila/n/nikita.saxena/activelearning/storage/grid/lower_quadrant_user_defined_context_fid{}_sanity.png".format(
    #             fid
    #         )
    #     )
    #     plt.close()
