from abc import ABC, abstractmethod
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
    DiscreteMaxValueBase,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    SampleReducingMCAcquisitionFunction,
)
from botorch.acquisition.cached_cholesky import CachedCholeskyMCSamplerMixin
from gflownet.utils.common import set_float_precision
from activelearning.surrogate.surrogate import Surrogate
from activelearning.dataset.dataset import DatasetHandler
from typing import Union
import torch
from botorch.models.gpytorch import GPyTorchModel


class Acquisition(ABC):
    def __init__(
        self,
        surrogate_model: any,  # TODO: define model type
        dataset_handler: DatasetHandler,
        device: Union[str, torch.device],
        float_precision: Union[int, torch.dtype],
        maximize=False,
    ) -> None:
        self.dataset_handler = dataset_handler
        self.device = device
        self.float = set_float_precision(float_precision)
        self.target_factor = 1 if maximize else -1
        self.maximize = maximize
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


class BOTorchMaxValueEntropyAcquisition(Acquisition):
    def __init__(
        self,
        surrogate_model: GPyTorchModel,
        acq_fn_class: DiscreteMaxValueBase = qLowerBoundMaxValueEntropy,
        dataset_handler=None,
        device="cpu",
        float_precision=64,
        maximize=False,
    ):
        super().__init__(
            surrogate_model, dataset_handler, device, float_precision, maximize
        )
        self.acq_fn_class = acq_fn_class

    def get_acquisition_values(self, candidate_set):
        candidate_set = candidate_set.to(self.device).to(self.float)
        self.acq_fn = self.acq_fn_class(
            self.surrogate_model,
            candidate_set=candidate_set,
        )
        candidate_set = candidate_set.to(self.device).to(self.float)
        return self.acq_fn(candidate_set.unsqueeze(1)).detach()


class BOTorchMonteCarloAcquisition(Acquisition):
    def __init__(
        self,
        surrogate_model: GPyTorchModel,
        acq_fn_class: SampleReducingMCAcquisitionFunction,
        dataset_handler: DatasetHandler,
        device="cpu",
        float_precision=64,
        maximize=False,
    ):
        super().__init__(
            surrogate_model, dataset_handler, device, float_precision, maximize
        )
        if issubclass(acq_fn_class, CachedCholeskyMCSamplerMixin):
            train_X, _ = dataset_handler.train_data[:]
            self.acq_fn = acq_fn_class(surrogate_model, train_X)
        else:
            best_f = dataset_handler.maxY() if self.maximize else dataset_handler.minY()
            self.acq_fn = acq_fn_class(surrogate_model, best_f * self.target_factor)

    def get_acquisition_values(self, candidate_set):
        candidate_set = candidate_set.to(self.device).to(self.float)
        return self.acq_fn(candidate_set.unsqueeze(1)).detach()