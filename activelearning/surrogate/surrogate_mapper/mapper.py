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
        self.batch_shape = model.batch_shape
        self._posterior_callback = None

    @abstractmethod
    def posterior(self, **kwargs) -> torch.Tensor:
        pass


class DifferenceMapper(SurrogateMapper):
    """
    states: Tensor of shape (batch, 1, no_features, 2)
    """

    def posterior(self, X: torch.Tensor, posterior_transform=None) -> GPyTorchPosterior:
        x1 = X[:, :, 0]
        x2 = X[:, :, 1]
        out1 = self.model.posterior(x1, posterior_transform=posterior_transform)
        out2 = self.model.posterior(x2, posterior_transform=posterior_transform)
        mean_diff = out1.mean - out2.mean
        covar_diff = (
            out1.covariance_matrix + out2.covariance_matrix
        )  # out1.covariance_matrix.abs() + out2.covariance_matrix.abs()

        # calculate euclidean distance to 1.6
        distance = (mean_diff - -1.6).abs()

        # -1 because smaller distances are better
        dist = MultivariateNormal(-1 * distance.squeeze(-1), covar_diff)
        posterior = GPyTorchPosterior(distribution=dist)

        # send results to a possible callback function
        if self._posterior_callback is not None:
            self._posterior_callback(posterior.mean, posterior.covariance_matrix)

        return posterior


class ConstantMapper(SurrogateMapper):
    """
    states: Tensor of shape (batch, 1, no_features)
    """

    def posterior(self, X: torch.Tensor, posterior_transform=None) -> GPyTorchPosterior:
        out = self.model.posterior(X, posterior_transform=posterior_transform)

        # calculate euclidean distance to 3.2
        distance = (out.mean.squeeze(-1) - -3.2).abs()

        # -1 because smaller distances are better
        dist = MultivariateNormal(-1 * distance, out.covariance_matrix)
        return GPyTorchPosterior(distribution=dist)
