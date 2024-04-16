from abc import ABC, abstractmethod
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
)
from gflownet.utils.common import set_float_precision
from activelearning.surrogate.surrogate import Surrogate
from typing import Union
import torch


class Acquisition(ABC):
    def __init__(
        self,
        surrogate_model: any,  # TODO: define model type
        device: Union[str, torch.device],
        float_precision: Union[int, torch.dtype],
        maximize=False,
    ) -> None:
        self.device = device
        self.float = set_float_precision(float_precision)
        self.target_factor = 1 if maximize else -1
        self.surrogate_model = surrogate_model

    def __call__(self, candidate_set: torch.Tensor) -> torch.Tensor:
        return (
            self.get_acquisition_values(candidate_set) * self.target_factor
        )  # TODO: remove self.target_factor; this will later be implemented directly in the gflownet environment (currently it always asumes negative values i.e. minimizing values)

    @abstractmethod
    def get_acquisition_values(self, candidate_set: torch.Tensor) -> torch.Tensor:
        pass

    def setup(self, env: any) -> None:
        # TODO: what is this needed for? has to be present for gflownet
        pass


botorch_acq_fn_mapper = {
    "qLowerBoundMaxValueEntropy": qLowerBoundMaxValueEntropy,
}


class BOTorchAcquisition(Acquisition):
    def __init__(
        self,
        surrogate_model,
        acq_fn_class="qLowerBoundMaxValueEntropy",
        device="cpu",
        float_precision=64,
        maximize=False,
    ):
        super().__init__(surrogate_model, device, float_precision, maximize)
        if type(acq_fn_class) == str:
            assert acq_fn_class in botorch_acq_fn_mapper, (
                "acq_fn_class should either be a function class or a string in "
                + botorch_acq_fn_mapper.keys()
            )
            acq_fn_class = botorch_acq_fn_mapper[acq_fn_class]
        self.acq_fn_class = acq_fn_class

    def get_acquisition_values(self, candidate_set):
        candidate_set = candidate_set.to(self.device).to(self.float)
        self.acq_fn = self.acq_fn_class(
            self.surrogate_model,
            candidate_set=candidate_set,
        )
        candidate_set = candidate_set.to(self.device).to(self.float)
        return self.acq_fn(candidate_set.unsqueeze(1)).detach()
