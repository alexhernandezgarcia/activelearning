import torch
import gpytorch
from tqdm import tqdm
import hydra
from botorch.models.gp_regression_fidelity import (
    SingleTaskMultiFidelityGP,
)
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll


# def unique(x, dim=-1):
#     unique, inverse = torch.unique(x, return_inverse=True, dim=dim)
#     perm = torch.arange(inverse.size(dim), dtype=inverse.dtype, device=inverse.device)
#     inverse, perm = inverse.flip([dim]), perm.flip([dim])
#     return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm)


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

        # train_x, index = unique(train_x, dim=0)
        # train_y = train_y[index]

        train_y = train_y * (-1)
        # samples, energies = self.dataset.shuffle(samples, energies)
        # train_x = samples[:self.n_samples, :-1].to(self.device)
        # targets = energies.to(self.device)
        # train_y = torch.stack([targets[i*self.n_samples:(i+1)*self.n_samples] for i in range(self.n_fid)], dim=-1)

        self.init_model(train_x, train_y)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        mll.to(train_x)
        mll = fit_gpytorch_mll(mll)
