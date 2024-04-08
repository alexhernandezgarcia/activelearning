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
        # self.maximize = maximize
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
    def get_predictions(self, states):
        pass

    def __call__(self, candidate_set):
        return (
            self.get_acquisition_values(candidate_set) * self.target_factor
        )  # TODO: remove self.target_factor; this will later be implemented directly in the gflownet environment (currently it always asumes negative values i.e. minimizing values)

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

    def get_predictions(self, states):
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
        candidate_set = candidate_set.to(self.device).to(self.float)
        acqf = qLowerBoundMaxValueEntropy(
            self.model,
            candidate_set=candidate_set,
        )
        return acqf(candidate_set.unsqueeze(1)).detach()
