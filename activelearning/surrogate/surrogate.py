from abc import abstractmethod, ABC
from gflownet.utils.common import set_float_precision
from typing import Optional, Union
import torch
from torch.utils.data import DataLoader
from activelearning.surrogate.surrogate_mapper.mapper import SurrogateMapper
from functools import partial


class Surrogate(ABC):
    def __init__(
        self,
        float_precision: Optional[Union[torch.dtype, int]] = 64,
        device: Optional[Union[str, torch.device]] = "cpu",
        surrogate_mapper_cls: partial[SurrogateMapper] = None,
        **kwargs
    ) -> None:
        # use the kwargs for model specific configuration that is implemented in subclasses
        self.float = set_float_precision(float_precision)
        self.device = device
        self.surrogate_mapper_cls = surrogate_mapper_cls

    @abstractmethod
    def fit(self, train_data: Union[torch.Tensor, DataLoader], **kwargs) -> None:
        # fit the surrogate model: self.model
        pass

    @abstractmethod
    def get_predictions(self, states: Union[torch.Tensor, DataLoader]) -> torch.Tensor:
        pass

    def get_model(self):
        if self.surrogate_mapper_cls is not None:
            return self.surrogate_mapper_cls(self.model)
        return self.model
