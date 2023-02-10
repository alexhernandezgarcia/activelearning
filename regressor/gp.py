import torch
import gpytorch
from tqdm import tqdm
import hydra
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll


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
        train_y = train["energies"].unsqueeze(-1) * (-1)
        # samples, energies = self.dataset.shuffle(samples, energies)
        # train_x = samples[:self.n_samples, :-1].to(self.device)
        # targets = energies.to(self.device)
        # train_y = torch.stack([targets[i*self.n_samples:(i+1)*self.n_samples] for i in range(self.n_fid)], dim=-1)

        self.init_model(train_x, train_y)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.model.likelihood, self.model
        )
        mll = fit_gpytorch_mll(mll)

        # print("Completed training of GP model? ", not mll.training)

        # self.model.train()
        # self.model.likelihood.train()

        # pbar = tqdm(range(1, self.max_iter + 1), disable=not self.progress)

        # for training_iter in pbar:
        #     self.optimizer.zero_grad()
        #     output = self.model(train_x)
        #     loss = -mll(output, train_y)
        #     loss.mean().backward()
        #     if self.progress:
        #         description = "Iter:{} | Train MLL: {:.4f}".format(
        #             training_iter, loss.mean().item()
        #         )
        #         pbar.set_description(description)
        #     self.optimizer.step()
