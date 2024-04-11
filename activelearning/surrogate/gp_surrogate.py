import torch
import gpytorch
from botorch.models.gp_regression_fidelity import (
    SingleTaskGP,
)
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.settings import debug

from activelearning.surrogate.surrogate import SurrogateModel
from activelearning.utils.helper_functions import dataloader_to_data


class GPSurrogate(SurrogateModel):
    # Gaussian Process Surrogate Model

    def _init_model(
        self, model_class, mll_class, likelihood=None, outcome_transform=None
    ):
        # initializes the model components for GP Surrogate Models
        # e.g.:
        #   model_class = botorch.models.gp_regression_fidelity.SingleTaskGP
        #   mll_class = gpytorch.mlls.ExactMarginalLogLikelihood # loss
        #   likelihood = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood
        #   outcome_transform = botorch.models.transforms.outcome.Standardize(m=1)
        self.model_class = model_class
        self.mll_class = mll_class
        self.likelihood = likelihood
        self.outcome_transform = outcome_transform

    def posterior(self, states, train=False):
        # returns the posterior of the surrogate model for the given states
        model = self.model

        if not train:
            model.eval()
            model.likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                posterior = model.posterior(states)
        else:
            model.train()
            model.likelihood.train()
            posterior = model.posterior(states)
        return posterior

    def fit(self, train_data):
        # fit the surrogate model
        train_x, train_y = dataloader_to_data(train_data)
        train_y = train_y.unsqueeze(-1).to(self.device).to(self.float)
        train_x = train_x.to(self.device).to(self.float)

        self.model = self.model_class(
            train_x,
            train_y * self.target_factor,
            outcome_transform=self.outcome_transform,
            likelihood=self.likelihood,
        )
        self.mll = self.mll_class(self.model.likelihood, self.model)
        self.mll.to(train_x)
        with debug(state=True):
            self.mll = fit_gpytorch_mll(self.mll)


class SingleTaskGPRegressor(GPSurrogate):
    # defines the SingleTaskGB as Surrogate
    def __init__(self, float_precision=64, device="cpu", maximize=False):
        super().__init__(
            float_precision,
            device,
            maximize,
            model_class=SingleTaskGP,
            mll_class=gpytorch.mlls.ExactMarginalLogLikelihood,
            likelihood=None,
            outcome_transform=Standardize(m=1),
        )
