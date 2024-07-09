import torch
from torch.utils.data import DataLoader
import gpytorch
from botorch.models.gp_regression_fidelity import SingleTaskGP
from botorch.models.approximate_gp import SingleTaskVariationalGP
from gpytorch.models.gp import GP
from botorch.models.transforms.outcome import Standardize, OutcomeTransform
from botorch.fit import fit_gpytorch_mll
from botorch.settings import debug
from botorch.models.model import Model

from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.helper_functions import dataloader_to_data
from typing import Optional, Union, Callable
import torch
from functools import partial

from gpytorch.constraints import GreaterThan
from torch.optim import SGD, Adam
from tqdm import tqdm
from activelearning.utils.logger import Logger
from activelearning.dataset.dataset import Data
from activelearning.utils.common import match_kwargs


class GPSurrogate(Surrogate):
    # Gaussian Process Surrogate Model

    def __init__(
        self,
        float_precision: Union[torch.dtype, int],
        device: Union[str, torch.device],
        model_class: partial[GP],
        mll_class: Optional[partial[gpytorch.mlls.MarginalLogLikelihood]] = None,
        likelihood: Optional[
            partial[gpytorch.likelihoods.likelihood._Likelihood]
        ] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        mll_args: dict = {},
        **kwargs: any,
    ) -> None:
        super().__init__(float_precision, device, **kwargs)
        # initializes the model components for GP Surrogate Models
        # e.g.:
        #   model_class = botorch.models.gp_regression_fidelity.SingleTaskGP
        #   mll_class = gpytorch.mlls.ExactMarginalLogLikelihood # loss
        #   likelihood = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood
        #   outcome_transform = botorch.models.transforms.outcome.Standardize(m=1)
        self.model_class = model_class
        if mll_class is None:
            mll_class = gpytorch.mlls.ExactMarginalLogLikelihood
        self.mll_class = mll_class
        self.likelihood = likelihood
        self.outcome_transform = outcome_transform
        self.kwargs = kwargs
        self.mll_args = mll_args

    def fit(self, train_data: Union[torch.Tensor, torch.utils.data.DataLoader]) -> None:
        # fit the surrogate model
        train_x, train_y = dataloader_to_data(train_data)
        train_y = train_y.to(self.device).to(self.float)
        train_x = train_x.to(self.device).to(self.float)

        self.model = self.model_class(
            train_x,
            train_y.unsqueeze(-1),
            outcome_transform=self.outcome_transform,
            likelihood=self.likelihood,
            **match_kwargs(self.kwargs, self.model_class),
        )
        gp_model = (
            self.model.model if hasattr(self.model, "model") else self.model
        )  # model.model needed for SingleTaskVariationalGP because it is a wrapper around the actual stochastic GP
        # gp_model = self.model

        self.mll = self.mll_class(
            self.model.likelihood,
            gp_model,
            **self.mll_args,
        )
        self.mll.to(train_x)
        with debug(state=True):
            self.mll = fit_gpytorch_mll(
                self.mll
            )  # for how many epochs does this function train? what optimizer does it use?

        ### alternative custom training: (see: https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html)
        ### tutorial from botorch: https://botorch.org/tutorials/fit_model_with_torch_optimizer
        # optimizer = torch.optim.Adam(...get parameters with self.model.parameters(),
        #     lr=0.01,
        # )
        # import tqdm
        # iterator = tqdm.notebook.tqdm(range(100))
        # for i in iterator:
        #     # Zero backprop gradients
        #     optimizer.zero_grad()
        #     # Get output from model
        #     output = self.model(train_x)
        #     # Calc loss and backprop derivatives
        #     loss = -self.mll(output, train_y)
        #     loss.backward()
        #     iterator.set_postfix(loss=loss.item())
        #     optimizer.step()

    def get_predictions(
        self, states: Union[torch.Tensor, Data, DataLoader]
    ) -> torch.Tensor:
        self.model.eval()
        self.model.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if isinstance(states, torch.utils.data.dataloader.DataLoader):
                y_mean = torch.Tensor([])
                y_std = torch.Tensor([])
                for batch in states:
                    posterior = self.model.posterior(
                        batch.to(self.device).to(self.float)
                    )
                    y_mean = torch.concat(
                        [
                            y_mean,
                            posterior.mean.cpu().squeeze(-1),
                        ]
                    )
                    y_std = torch.concat(
                        [
                            y_std,
                            posterior.variance.cpu().sqrt().squeeze(-1),
                        ]
                    )
                return torch.concat([y_mean.unsqueeze(0), y_std.unsqueeze(0)], dim=0)
            else:
                states_proxy = states[:].to(self.device).to(self.float)
                posterior = self.model.posterior(states_proxy)
                y_mean = posterior.mean
                y_std = posterior.variance.sqrt()
                y_mean = y_mean.squeeze(-1).unsqueeze(0)
                y_std = y_std.squeeze(-1).unsqueeze(0)
                return torch.concat([y_mean, y_std], dim=0)


class SingleTaskGPRegressor(GPSurrogate):
    # defines the SingleTaskGP as Surrogate
    def __init__(self, float_precision, device, **kwargs):
        super().__init__(
            model_class=SingleTaskGP,
            mll_class=gpytorch.mlls.ExactMarginalLogLikelihood,
            likelihood=None,
            outcome_transform=Standardize(m=1),
            float_precision=float_precision,
            device=device,
        )


class SVGPSurrogate(GPSurrogate):
    def __init__(
        self,
        float_precision,
        device,
        model_class=SingleTaskVariationalGP,
        mll_class=gpytorch.mlls.VariationalELBO,
        likelihood=None,
        outcome_transform=Standardize(m=1),
        mll_args: dict = {},
        train_epochs: int = 150,
        lr: float = 0.1,
        logger: Logger = None,
        id: str = "",
        **kwargs: any,
    ) -> None:
        super().__init__(
            model_class=model_class,
            mll_class=mll_class,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            float_precision=float_precision,
            device=device,
            mll_args=mll_args,
            **kwargs,
        )
        self.train_epochs = train_epochs
        self.lr = lr
        self.logger = logger

    def fit(self, train_data: torch.utils.data.DataLoader) -> None:
        # fit the surrogate model
        batch_x, batch_y = next(iter(train_data))
        batch_x = batch_x.to(self.device).to(self.float)
        batch_y = batch_y
        batch_y = batch_y.to(self.device).to(
            self.float
        )  # turn into maximization problem

        self.model = self.model_class(
            batch_x,
            batch_y.unsqueeze(-1),
            outcome_transform=self.outcome_transform,
            likelihood=self.likelihood,
            **match_kwargs(self.kwargs, self.model_class),
        )
        self.model.likelihood.noise_covar.register_constraint(
            "raw_noise", GreaterThan(1e-5)
        )

        gp_model = (
            self.model.model if hasattr(self.model, "model") else self.model
        )  # model.model needed for SingleTaskVariationalGP because it is a wrapper around the actual stochastic GP
        # gp_model = self.model

        self.mll = self.mll_class(
            self.model.likelihood,
            gp_model,
            **self.mll_args,
        )
        self.mll.to(batch_x)

        optimizer = Adam([{"params": self.model.parameters()}], lr=self.lr)
        self.model.train()

        epochs_iter = tqdm(range(self.train_epochs), desc="Epoch")
        avg_losses = []
        for epoch in epochs_iter:
            batch_losses = []
            for x_batch, y_batch in train_data:
                x_batch = x_batch.to(self.device).to(self.float)
                y_batch = y_batch.to(self.device).to(
                    self.float
                )  # turn into maximization problem
                optimizer.zero_grad()
                output = self.model(x_batch)
                loss = -self.mll(output, y_batch)
                epochs_iter.set_postfix(loss=loss.item())
                batch_losses.append(loss.item())
                loss.backward()
                optimizer.step()

            avg_losses.append(sum(batch_losses) / len(batch_losses))

        if self.logger is not None:
            self.logger.log_time_series(avg_losses, "avg_loss_surrogate")


from activelearning.surrogate.gp_kernels import (
    DeepKernelConstantMean,
    DeepKernelWrapper,
)
from botorch.models.utils.gpytorch_modules import (
    get_matern_kernel_with_gamma_prior,
)
from functools import partial


class DeepKernelSVGPSurrogate(SVGPSurrogate):
    def __init__(
        self,
        feature_extractor,
        float_precision,
        device,
        mll_args: dict = {},
        train_epochs: int = 150,
        lr: float = 0.1,
        logger: Logger = None,
        **kwargs: any,
    ):
        covar_module = DeepKernelWrapper(
            get_matern_kernel_with_gamma_prior(
                ard_num_dims=feature_extractor.n_output,
                batch_shape=torch.Size(),
            ),
            feature_extractor,
        )
        mean_module = DeepKernelConstantMean(
            feature_extractor, batch_shape=torch.Size()
        )
        super().__init__(
            model_class=partial(
                SingleTaskVariationalGP,
                covar_module=covar_module,
                mean_module=mean_module,
            ),
            float_precision=float_precision,
            device=device,
            mll_args=mll_args,
            train_epochs=train_epochs,
            lr=lr,
            logger=logger,
            **kwargs,
        )
