from botorch.models.approximate_gp import ApproximateGP, ApproximateGPyTorchModel
from typing import Optional, Type, Union
from torch import Tensor
from gpytorch.variational import (
    _VariationalDistribution,
    _VariationalStrategy,
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)
from gpytorch.kernels import Kernel, MaternKernel, IndexKernel, ScaleKernel
from gpytorch.likelihoods import (
    GaussianLikelihood,
    Likelihood,
    MultitaskGaussianLikelihood,
)
from gpytorch.means import ConstantMean, Mean
import copy
import torch
from gpytorch.priors import GammaPrior

# from botorch.models.approximate_gp import _select_inducing_points
from botorch.models.approximate_gp import _pivoted_cholesky_init
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.transforms.input import InputTransform
from botorch.models.utils import validate_input_scaling
from gpytorch.constraints import GreaterThan
from gpytorch.utils.memoize import clear_cache_hook
from gpytorch.module import Module
from gpytorch.distributions import MultivariateNormal

MIN_INFERRED_NOISE_LEVEL = 1e-4

# TODO: Adopt inheritance
class SingleTaskMultiFidelityVariationalGP(ApproximateGPyTorchModel):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Optional[Tensor] = None,
        likelihood: Optional[Likelihood] = None,
        num_outputs: int = 1,
        learn_inducing_points: bool = True,
        covar_module_x: Optional[Kernel] = None,
        covar_module_fidelity: Optional[Kernel] = None,
        mean_module: Optional[Mean] = None,
        variational_distribution: Optional[_VariationalDistribution] = None,
        variational_strategy: Type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Optional[Union[Tensor, int]] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
    ) -> None:
        with torch.no_grad():
            transformed_X = self.transform_inputs(
                X=train_X, input_transform=input_transform
            )
        if train_Y is not None:
            if outcome_transform is not None:
                train_Y, _ = outcome_transform(train_Y)
            self._validate_tensor_args(X=transformed_X, Y=train_Y)
            validate_input_scaling(train_X=transformed_X, train_Y=train_Y)
            if train_Y.shape[-1] != num_outputs:
                num_outputs = train_Y.shape[-1]

        self._num_outputs = num_outputs
        self._input_batch_shape = train_X.shape[:-2]
        aug_batch_shape = copy.deepcopy(self._input_batch_shape)
        if num_outputs > 1:
            aug_batch_shape += torch.Size([num_outputs])
        self._aug_batch_shape = aug_batch_shape

        if likelihood is None:
            raise NotImplementedError(
                "Likelihood must be provided while initalising user-defined SingleTaskMultiFidelityVariationalGP."
            )
            if num_outputs == 1:
                noise_prior = GammaPrior(1.1, 0.05)
                noise_prior_mode = (noise_prior.concentration - 1) / noise_prior.rate
                likelihood = GaussianLikelihood(
                    noise_prior=noise_prior,
                    batch_shape=self._aug_batch_shape,
                    noise_constraint=GreaterThan(
                        MIN_INFERRED_NOISE_LEVEL,
                        transform=None,
                        initial_value=noise_prior_mode,
                    ),
                )
            else:
                raise NotImplementedError(
                    "Multitask likelihood should not be used in SingleTaskMultiFidelityVariationalGP."
                )
                likelihood = MultitaskGaussianLikelihood(num_tasks=num_outputs)
        else:
            self._is_custom_likelihood = True

        model = _SingleTaskMultiFidelityVariationalGP(
            train_X=transformed_X,
            train_Y=train_Y,
            num_outputs=num_outputs,
            learn_inducing_points=learn_inducing_points,
            covar_module_x=covar_module_x,
            covar_module_fidelity=covar_module_fidelity,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
        )

        super().__init__(model=model, likelihood=likelihood, num_outputs=num_outputs)

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform
        if input_transform is not None:
            self.input_transform = input_transform

        # for model fitting utilities
        # TODO: make this a flag?
        self.model.train_inputs = [transformed_X]
        if train_Y is not None:
            self.model.train_targets = train_Y.squeeze(-1)

        self.to(train_X)

    def init_inducing_points(
        self,
        inputs: Tensor,
    ) -> Tensor:
        r"""
        Reinitialize the inducing point locations in-place with the current kernel
        applied to `inputs`.
        The variational distribution and variational strategy caches are reset.
        Args:
            inputs: (\*batch_shape, n, d)-dim input data tensor.
        Returns:
            (\*batch_shape, m, d)-dim tensor of selected inducing point locations.
        """
        var_strat = self.model.variational_strategy
        clear_cache_hook(var_strat)
        if hasattr(var_strat, "base_variational_strategy"):
            var_strat = var_strat.base_variational_strategy
            clear_cache_hook(var_strat)

        with torch.no_grad():
            num_inducing = var_strat.inducing_points.size(-2)
            inducing_points = _select_inducing_points(
                inputs=inputs,
                covar_module_x=self.model.covar_module_x,
                covar_module_fidelity=self.model.covar_module_fidelity,
                num_inducing=num_inducing,
                input_batch_shape=self._input_batch_shape,
            )
            var_strat.inducing_points.copy_(inducing_points)
            var_strat.variational_params_initialized.fill_(0)

        return inducing_points


def _select_inducing_points(
    inputs: Tensor,
    covar_module_x: Module,
    covar_module_fidelity: Module,
    num_inducing: int,
    input_batch_shape: torch.Size,
) -> Tensor:
    r"""
    Utility function that evaluates a kernel at given inputs and selects inducing point
    locations based on the pivoted Cholesky heuristic.
    Args:
        inputs: A (*batch_shape, n, d)-dim input data tensor.
        covar_module: GPyTorch Module returning a LinearOperator kernel matrix.
        num_inducing: The maximun number (m) of inducing points (m <= n).
        input_batch_shape: The non-task-related batch shape.
    Returns:
        A (*batch_shape, m, d)-dim tensor of inducing point locations.
    """
    x_input = inputs[..., :-1]
    f_input = inputs[..., -1].long()
    x_output = covar_module_x(x_input)
    f_output = covar_module_fidelity(f_input)
    output = x_output.mul(f_output)
    train_train_kernel = output.evaluate_kernel()
    # train_train_kernel = x_output.evaluate_kernel() + f_output.evaluate_kernel()
    # train_train_kernel = covar_module(inputs).evaluate_kernel()

    # base case
    if train_train_kernel.ndimension() == 2:
        inducing_points = _pivoted_cholesky_init(
            train_inputs=inputs,
            kernel_matrix=train_train_kernel,
            max_length=num_inducing,
        )
    # multi-task case
    elif train_train_kernel.ndimension() == 3 and len(input_batch_shape) == 0:
        raise NotImplementedError(
            "UserDefinedError: Multi-task cases not yet supported by SingleTaskMultiFidelityGP."
        )
        # input_element = inputs[0] if inputs.ndimension() == 3 else inputs
        # kernel_element = train_train_kernel[0]
        # inducing_points = _pivoted_cholesky_init(
        #     train_inputs=input_element,
        #     kernel_matrix=kernel_element,
        #     max_length=num_inducing,
        # )
    # batched input cases
    else:
        raise NotImplementedError(
            "UserDefinedError: Batched input cases not yet supported by SingleTaskMultiFidelityGP."
        )
        # batched_inputs = (
        #     inputs.expand(*input_batch_shape, -1, -1)
        #     if inputs.ndimension() == 2
        #     else inputs
        # )
        # reshaped_inputs = batched_inputs.flatten(end_dim=-3)
        # inducing_points = []
        # for input_element in reshaped_inputs:
        #     # the extra kernel evals are a little wasteful but make it
        #     # easier to infer the task batch size
        #     kernel_element = covar_module(input_element).evaluate_kernel()
        #     # handle extra task batch dimension
        #     kernel_element = (
        #         kernel_element[0]
        #         if kernel_element.ndimension() == 3
        #         else kernel_element
        #     )
        #     inducing_points.append(
        #         _pivoted_cholesky_init(
        #             train_inputs=input_element,
        #             kernel_matrix=kernel_element,
        #             max_length=num_inducing,
        #         )
        #     )
        # inducing_points = torch.stack(inducing_points).view(
        #     *input_batch_shape, num_inducing, -1
        # )

    return inducing_points


class _SingleTaskMultiFidelityVariationalGP(ApproximateGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Optional[Tensor] = None,
        num_outputs: int = 1,
        learn_inducing_points=True,
        covar_module_x: Optional[Kernel] = None,
        covar_module_fidelity: Optional[Kernel] = None,
        mean_module: Optional[Mean] = None,
        variational_distribution: Optional[_VariationalDistribution] = None,
        variational_strategy: Type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Optional[Union[Tensor, int]] = None,
        num_fidelity: int = 1,
    ) -> None:
        input_batch_shape = train_X.shape[:-2]
        aug_batch_shape = copy.deepcopy(input_batch_shape)
        if num_outputs > 1:
            aug_batch_shape += torch.Size((num_outputs,))
        self._aug_batch_shape = aug_batch_shape

        if mean_module is None:
            mean_module = ConstantMean(batch_shape=self._aug_batch_shape).to(train_X)

        if covar_module_x is None or covar_module_fidelity is None:
            raise NotImplementedError(
                "UserDefinedError: Covariance module of x and fidelity must be provided."
            )
            covar_module = (
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=train_X.shape[-1],
                    batch_shape=self._aug_batch_shape,
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                ).to(train_X),
            )
            # batch_shape=self._aug_batch_shape,
            # outputscale_prior=GammaPrior(2.0, 0.15),
            # )
            self._subset_batch_dict = {
                "mean_module.constant": -2,
                "covar_module.raw_outputscale": -1,
                "covar_module.base_kernel.raw_lengthscale": -3,
            }
            task_covar_module = IndexKernel(num_tasks=num_fidelity, rank=1)

        # initialize inducing points with a pivoted cholesky init if they are not given
        if not isinstance(inducing_points, Tensor):
            raise NotImplementedError(
                "UserDefinedError: Inducing points must be provided."
            )
            if inducing_points is None:
                # number of inducing points is 25% the number of data points
                # as a heuristic
                inducing_points = int(0.25 * train_X.shape[-2])

            inducing_points = _select_inducing_points(
                inputs=train_X,
                covar_module_x=covar_module_x,
                covar_module_fidelity=covar_module_fidelity,
                num_inducing=inducing_points,
                input_batch_shape=input_batch_shape,
            )

        if variational_distribution is None:
            variational_distribution = CholeskyVariationalDistribution(
                num_inducing_points=inducing_points.shape[-2],
                batch_shape=self._aug_batch_shape,
            )

        variational_strategy = variational_strategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=learn_inducing_points,
        )

        # wrap variational models in independent multi-task variational strategy
        if num_outputs > 1:
            raise NotImplementedError(
                "UserDefinedError: Multi-output in _SingleTaskMultiFidelityVariationalGP not yet supported."
            )
            variational_strategy = IndependentMultitaskVariationalStrategy(
                base_variational_strategy=variational_strategy,
                num_tasks=num_outputs,
                task_dim=-1,
            )
        super().__init__(variational_strategy=variational_strategy)
        self.mean_module = mean_module
        self.covar_module_x = covar_module_x
        self.covar_module_fidelity = covar_module_fidelity

    def forward(self, input) -> MultivariateNormal:
        i = input[..., -1].long()
        x = input[..., :-1]
        mean_x = self.mean_module(x)
        covar_x = self.covar_module_x(x)
        covar_fidelity = self.covar_module_fidelity(i)
        covar = covar_x.mul(covar_fidelity)
        latent_dist = MultivariateNormal(mean_x, covar)
        return latent_dist
