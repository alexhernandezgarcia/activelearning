from botorch.models.gp_regression_fidelity import SingleTaskGP
from botorch.models.utils.gpytorch_modules import (
    MIN_INFERRED_NOISE_LEVEL,
    get_gaussian_likelihood_with_gamma_prior,
    get_matern_kernel_with_gamma_prior,
)


class SingleTaskDKLModel(SingleTaskGP):
    # This model is a wrapper that connects a feature extractor
    # (i.e. any kind of pytorch model) with the SingleTaskGP botorch model

    # DKL code adapted from:
    # https://docs.gpytorch.ai/en/stable/examples/06_PyTorch_NN_Integration_DKL/KISSGP_Deep_Kernel_Regression_CUDA.html
    # https://docs.gpytorch.ai/en/stable/examples/06_PyTorch_NN_Integration_DKL/Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.html
    def __init__(
        self, train_x, train_y, feature_extractor, likelihood, outcome_transform
    ):
        covar_module = get_matern_kernel_with_gamma_prior(
            ard_num_dims=feature_extractor.n_output,
            batch_shape=None,
        )
        super(SingleTaskDKLModel, self).__init__(
            train_x,
            train_y,
            likelihood=likelihood,
            outcome_transform=outcome_transform,
            covar_module=covar_module,
        )
        self.feature_extractor = feature_extractor

        # This module will scale the NN features so that they're nice values
        # self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)
        # projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"
        return super().forward(projected_x)
