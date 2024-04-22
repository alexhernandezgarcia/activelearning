import torch
import gpytorch
from botorch.models.gp_regression_fidelity import SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.transforms.outcome import Standardize, OutcomeTransform
from botorch.fit import fit_gpytorch_mll
from botorch.settings import debug
from botorch.models.model import Model

from activelearning.surrogate.surrogate import Surrogate
from activelearning.utils.helper_functions import dataloader_to_data
from typing import Optional, Union
import torch
from functools import partial


class GPSurrogate(Surrogate):
    # Gaussian Process Surrogate Model

    def __init__(
        self,
        float_precision: Union[torch.dtype, int],
        device: Union[str, torch.device],
        model_class: partial[GPyTorchModel],
        mll_class: Optional[partial[gpytorch.mlls.MarginalLogLikelihood]] = None,
        likelihood: Optional[
            partial[gpytorch.likelihoods.likelihood._Likelihood]
        ] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        maximize: bool = False,
        mll_args: dict = {},
        **kwargs: any
    ) -> None:
        super().__init__(float_precision, device, maximize)
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
            train_y.unsqueeze(-1) * self.target_factor,
            outcome_transform=self.outcome_transform,
            likelihood=self.likelihood,
            **self.kwargs,
        )
        gp_model = self.model.model if hasattr(self.model, "model") else self.model
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

    def get_predictions(self, states: torch.Tensor) -> torch.Tensor:
        states_proxy = states.clone().to(self.device).to(self.float)
        self.model.eval()
        self.model.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self.model.posterior(states_proxy)
        y_mean = posterior.mean
        y_std = posterior.variance.sqrt()
        y_mean = y_mean.squeeze(-1).unsqueeze(0)
        y_std = y_std.squeeze(-1).unsqueeze(0)
        return torch.concat([y_mean, y_std], dim=0)


class SingleTaskGPRegressor(GPSurrogate):
    # defines the SingleTaskGP as Surrogate
    def __init__(self, float_precision, device, maximize=False):
        super().__init__(
            model_class=SingleTaskGP,
            mll_class=gpytorch.mlls.ExactMarginalLogLikelihood,
            likelihood=None,
            outcome_transform=Standardize(m=1),
            float_precision=float_precision,
            device=device,
            maximize=maximize,
        )
