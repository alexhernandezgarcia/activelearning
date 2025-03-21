from abc import ABC, abstractmethod
from typing import Optional, Union

import torch
from gflownet.utils.common import set_float_precision
from torch.utils.data import DataLoader


class Surrogate(ABC):
    def __init__(
        self,
        float_precision: Optional[Union[torch.dtype, int]] = 64,
        device: Optional[Union[str, torch.device]] = "cpu",
        **kwargs,
    ) -> None:
        # use the kwargs for model specific configuration that is implemented in subclasses
        self.float = set_float_precision(float_precision)
        self.device = device

    @abstractmethod
    def fit(self, train_data: Union[torch.Tensor, DataLoader]) -> None:
        # fit the surrogate model: self.model
        pass

    @abstractmethod
    def get_predictions(self, states: Union[torch.Tensor, DataLoader]) -> torch.Tensor:
        pass
