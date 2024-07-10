from abc import ABC, abstractmethod
from typing import Any
import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal


class SurrogateMapper(ABC):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        self.num_outputs = model.num_outputs

    @abstractmethod
    def posterior(self, **kwargs) -> torch.Tensor:
        pass


class DifferenceMapper(SurrogateMapper):
    """
    states: Tensor of shape (batch, 1, no_features, 2)
    """

    def posterior(self, X: torch.Tensor, posterior_transform=None) -> GPyTorchPosterior:
        x1 = X[:, :, :, 0]
        x2 = X[:, :, :, 1]
        out1 = self.model.posterior(x1, posterior_transform=posterior_transform)
        out2 = self.model.posterior(x2, posterior_transform=posterior_transform)
        mean_diff = out1.mean - out2.mean
        covar_diff = out1.covariance_matrix.abs() + out2.covariance_matrix.abs()
        dist = MultivariateNormal(mean_diff.squeeze(-1), covar_diff)
        return GPyTorchPosterior(distribution=dist)


class ConstantMapper(SurrogateMapper):
    """
    states: Tensor of shape (batch, 1, no_features)
    """

    def posterior(self, X: torch.Tensor, posterior_transform=None) -> GPyTorchPosterior:
        out = self.model.posterior(X, posterior_transform=posterior_transform)
        dist = MultivariateNormal(out.mean.squeeze(-1) - 3.2, out.covariance_matrix)
        return GPyTorchPosterior(distribution=dist)
