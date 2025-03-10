import torch
from gpytorch.kernels import Kernel
from gpytorch.means import ConstantMean


class DeepKernelConstantMean(ConstantMean):
    def __init__(
        self,
        feature_extractor,
        constant_prior=None,
        constant_constraint=None,
        batch_shape=torch.Size(),
        **kwargs,
    ):
        super(DeepKernelConstantMean, self).__init__(
            constant_prior, constant_constraint, batch_shape, kwargs=kwargs
        )
        self.feature_extractor = feature_extractor

    def forward(self, input):
        input = self.feature_extractor(input)
        return super(DeepKernelConstantMean, self).forward(input)


class DeepKernelWrapper(Kernel):
    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if base kernel is stationary.
        """
        return self.base_kernel.is_stationary

    def __init__(
        self,
        base_kernel: Kernel,
        feature_extractor,
        **kwargs,
    ):
        if hasattr(base_kernel, "active_dims") and base_kernel.active_dims is not None:
            kwargs["active_dims"] = base_kernel.active_dims
        super(DeepKernelWrapper, self).__init__(**kwargs)

        self.base_kernel = base_kernel
        self.feature_extractor = feature_extractor

    def forward(self, x1, x2=None, last_dim_is_batch=False, diag=False, **params):
        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)
        base_kernel_output = self.base_kernel.forward(
            x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params
        )
        return base_kernel_output

    def num_outputs_per_input(self, x1, x2):
        return self.base_kernel.num_outputs_per_input(x1, x2)

    def prediction_strategy(
        self, train_inputs, train_prior_dist, train_labels, likelihood
    ):
        return self.base_kernel.prediction_strategy(
            train_inputs, train_prior_dist, train_labels, likelihood
        )
