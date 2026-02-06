from torch import Tensor
from typing import Optional
from botorch.acquisition.objective import MCAcquisitionObjective


class IdentityMCObjective(MCAcquisitionObjective):
    r"""Trivial objective extracting the last dimension.

    Example:
        >>> identity_objective = IdentityMCObjective()
        >>> samples = sampler(posterior)
        >>> objective = identity_objective(samples)
    """

    _verify_output_shape = False

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        # return samples.squeeze(-1)
        return samples
