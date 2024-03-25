import torch
import gpytorch
from botorch.models.gp_regression_fidelity import (
    SingleTaskGP,
)
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
)
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
import numpy as np
from abc import abstractmethod, ABC
from botorch.settings import debug
from gflownet.utils.common import set_float_precision


class Surrogate(ABC):
    def __init__(self, float_precision=64, device="cpu", maximize=False):
        self.maximize = maximize
        self.target_factor = 1 if maximize else -1
        self.float = set_float_precision(float_precision)
        self.device = device

    @abstractmethod
    def fit(self, train_data):
        # train_data is a pytorch dataloader
        pass

    # TODO: what is this method for? needed by Environment
    def setup(self, env):
        pass

    @abstractmethod
    def __call__(self, states):
        pass

    @abstractmethod
    def get_acquisition_values(self, candidate_set):
        pass


class SingleTaskGPRegressor(Surrogate):

    def dataloader_to_data(self, train_data):
        # TODO: check if there is a better way to use dataloaders with botorch
        train_x = torch.Tensor()
        train_y = torch.Tensor()
        for state, score in train_data:
            train_x = torch.cat((train_x, state), 0)
            train_y = torch.cat((train_y, score), 0)

        return train_x, train_y

    def fit(self, train_data):
        train_x, train_y = self.dataloader_to_data(train_data)
        train_y = train_y.unsqueeze(-1).to(self.device).to(self.float)
        train_x = train_x.to(self.device).to(self.float)

        self.model = SingleTaskGP(
            train_x,
            train_y * self.target_factor,
            outcome_transform=Standardize(m=1),
        )
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        self.mll.to(train_x)
        with debug(state=True):
            self.mll = fit_gpytorch_mll(self.mll)

    def __call__(self, states):
        """Input is states
        Proxy conversion happens within."""
        states_proxy = states.clone().to(self.device).to(self.float)
        model = self.model
        model.eval()
        model.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = model.posterior(states_proxy)
            y_mean = posterior.mean
            y_std = posterior.variance.sqrt()
        y_mean = y_mean.squeeze(-1)
        y_std = y_std.squeeze(-1)
        return y_mean, y_std

    def get_acquisition_values(self, candidate_set):
        acqf = qLowerBoundMaxValueEntropy(
            self.model,
            candidate_set=candidate_set,
        )
        return acqf(candidate_set.unsqueeze(1))

    # def get_metrics(self, y_mean, y_std, env, states):
    #     state_oracle_input = states.clone()
    #     if hasattr(env, "call_oracle_per_fidelity"):
    #         samples, fidelity = env.statebatch2oracle(state_oracle_input)
    #         targets = env.call_oracle_per_fidelity(samples, fidelity).detach().cpu()
    #     elif hasattr(env, "oracle"):
    #         samples = env.statebatch2oracle(state_oracle_input)
    #         targets = env.oracle(samples).detach().cpu()
    #     targets = targets * self.target_factor
    #     targets_numpy = targets.detach().cpu().numpy()
    #     targets_numpy = targets_numpy
    #     rmse = np.sqrt(np.mean((y_mean - targets_numpy) ** 2))
    #     nll = (
    #         -torch.distributions.Normal(torch.tensor(y_mean), torch.tensor(y_std))
    #         .log_prob(targets)
    #         .mean()
    #     )
    #     return rmse, nll

    # def get_modes(self, states, env):
    #     num_pick = int((env.length * env.length) / 100) * 5
    #     percent = num_pick / (env.length * env.length) * 100
    #     print(
    #         "\nUser-Defined Warning: Top {}% of states with maximum reward are picked for GP mode metrics".format(
    #             percent
    #         )
    #     )
    #     state_oracle_input = states.clone()
    #     if hasattr(env, "call_oracle_per_fidelity"):
    #         samples, fidelity = env.statebatch2oracle(state_oracle_input)
    #         targets = env.oracle[-1](samples).detach().cpu()
    #     elif hasattr(env, "oracle"):
    #         samples = env.statebatch2oracle(state_oracle_input)
    #         targets = env.oracle(samples).detach().cpu()
    #     targets = targets * self.target_factor
    #     idx_pick = torch.argsort(targets, descending=True)[:num_pick].tolist()
    #     states_pick = states[idx_pick]
    #     if hasattr(env, "oracle"):
    #         self._mode = states_pick
    #     else:
    #         fidelities = torch.zeros((len(states_pick) * 3, 1)).to(states_pick.device)
    #         for i in range(self.n_fid):
    #             fidelities[i * len(states_pick) : (i + 1) * len(states_pick), 0] = i
    #         states_pick = states_pick.repeat(self.n_fid, 1)
    #         state_pick_fid = torch.cat([states_pick, fidelities], dim=1)
    #         self._mode = state_pick_fid

    # def evaluate_model(self, env, do_figure=True):
    #     if env.n_dim > 2:
    #         return None, 0.0, 0.0, 0.0, 0.0
    #     states = torch.FloatTensor(env.get_all_terminating_states()).to("cuda")
    #     y_mean, y_std = self.get_predictions(env, states)
    #     rmse, nll = self.get_metrics(y_mean, y_std, env, states)
    #     if hasattr(self, "_mode") == False:
    #         self.get_modes(states, env)
    #     mode_mean, mode_std = self.get_predictions(env, self._mode)
    #     mode_rmse, mode_nll = self.get_metrics(mode_mean, mode_std, env, self._mode)
    #     if do_figure:
    #         figure1 = self.plot_predictions(states, y_mean, env.length, env.rescale)
    #         figure2 = self.plot_predictions(states, y_std, env.length, env.rescale)
    #         fig = [figure1, figure2]
    #     else:
    #         fig = None
    #     return fig, rmse, nll, mode_rmse, mode_nll
