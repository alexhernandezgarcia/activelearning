import torch
from botorch.models.model import Model
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from botorch.posteriors.gpytorch import GPyTorchPosterior
import os


class ProxyBotorchUCB(Model):
    def __init__(self, regressor, num_dropout_samples):
        super().__init__()
        self.regressor = regressor
        self._num_outputs = 1
        self.num_dropout_samples = num_dropout_samples

    def posterior(self, X, observation_noise=False, posterior_transform=None):
        super().posterior(X, observation_noise, posterior_transform)

        self.regressor.model.train()
        dim = X.ndim
        if dim == 3:
            X = X.squeeze(-2)

        with torch.no_grad():
            outputs = self.regressor.forward_with_uncertainty(
                X, self.num_dropout_samples
            )
        mean = torch.mean(outputs, dim=1).unsqueeze(-1)
        var = torch.var(outputs, dim=1).unsqueeze(-1)
        # if var is an array of zeros then we add a small value to it
        var = torch.where(var == 0, torch.ones_like(var) * 1e-4, var)

        covar = [torch.diag(var[i]) for i in range(X.shape[0])]
        covar = torch.stack(covar, axis=0)
        mvn = MultivariateNormal(mean, covar)
        posterior = GPyTorchPosterior(mvn)
        return posterior

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self):
        """
        This is a batch shape from an I/O perspective. For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        return torch.Size([])


class MultifidelityOracleModel(Model):
    def __init__(self, oracle, n_fid):
        super().__init__()
        self.oracle = oracle
        self._num_outputs = 1
        self.n_fid = n_fid

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    @property
    def batch_shape(self):
        """
        This is a batch shape from an I/O perspective. For a model with `m` outputs, a `test_batch_shape x q x d`-shaped input `X`
        to the `posterior` method returns a Posterior object over an output of
        shape `broadcast(test_batch_shape, model.batch_shape) x q x m`.
        """
        return torch.Size([])

    def posterior(self, X, observation_noise=False, posterior_transform=None):
        # assumes mfenv.statetorch2oracle is called
        super().posterior(X, observation_noise, posterior_transform)
        fid_tensor = X[:, -1]
        state_tensor = X[:, :-1]

        mean = torch.zeros(X.shape[0], 1)
        var = torch.zeros(X.shape[0], 1)

        for fid in range(self.n_fid):
            idx_fid = torch.where(fid_tensor == fid)[0]
            # state_fid = state_tensor[idx_fid]
            mean[idx_fid] = self.oracle[fid].mu
            var[idx_fid] = self.oracle[fid].sigma ** 2

        if len(X.shape) == 2:
            mean = mean.squeeze(-1)
            var = var.squeeze(-1)
            covar = torch.diag(var)
        elif len(X.shape) == 4:
            mean = mean.squeeze(-1).unsqueeze(1)
            # construct covariance matrix from mean and var
            covar = [torch.diag(var[i]) for i in range(X.shape[0])]
            covar = torch.stack(covar, axis=0)
            covar = covar.unsqueeze(1)
        elif len(X.shape) == 3:
            covar = [torch.diag(var[i]) for i in range(X.shape[0])]
            covar = torch.stack(covar, axis=0)

        mvn = MultivariateNormal(mean, covar)
        posterior = GPyTorchPosterior(mvn)
        return posterior
