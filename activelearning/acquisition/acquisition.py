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
from functools import partial


class Acquisition(ABC):
    def __init__(
        self,
        surrogate_model: any,  # TODO: define model type
        dataset_handler: DatasetHandler,
        device: Union[str, torch.device],
        float_precision: Union[int, torch.dtype],
    ) -> None:
        self.dataset_handler = dataset_handler
        self.device = device
        self.float = set_float_precision(float_precision)
        self.surrogate_model = surrogate_model

    def __call__(self, candidate_set: torch.Tensor) -> torch.Tensor:
        """
        Evaluates the acquisition function on a set of candidates.
        """
        if isinstance(candidate_set, torch.utils.data.dataloader.DataLoader):
            values = torch.Tensor([])
            for batch in candidate_set:
                values = torch.concat(
                    [
                        values,
                        self.get_acquisition_values(batch).cpu(),
                    ]
                )
            return values
        else:
            return self.get_acquisition_values(candidate_set)

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
    ):
        super().__init__(surrogate_model, dataset_handler, device, float_precision)
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
    ):
        super().__init__(surrogate_model, dataset_handler, device, float_precision)
        base_class = (
            acq_fn_class.func if isinstance(acq_fn_class, partial) else acq_fn_class
        )
        if issubclass(base_class, CachedCholeskyMCSamplerMixin):
            train_X, _ = dataset_handler.train_data[:]
            self.acq_fn = acq_fn_class(
                surrogate_model,
                train_X.to(self.device).to(self.float),
            )
        else:
            best_f = dataset_handler.maxY()
            self.acq_fn = acq_fn_class(
                surrogate_model,
                best_f,
            )

    def get_acquisition_values(self, candidate_set):
        candidate_set = candidate_set.to(self.device).to(self.float)
        return self.acq_fn(candidate_set.unsqueeze(1)).detach()
