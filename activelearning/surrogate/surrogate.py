from abc import abstractmethod, ABC
from gflownet.utils.common import set_float_precision
from typing import Optional, Union
import torch
from torch.utils.data import DataLoader


class Surrogate(ABC):
    def __init__(
        self,
        float_precision: Optional[Union[torch.dtype, int]] = 64,
        device: Optional[Union[str, torch.device]] = "cpu",
        maximize: bool = False,
        **kwargs
    ) -> None:
        # use the kwargs for model specific configuration that is implemented in subclasses
        self.maximize = maximize
        self.target_factor = 1 if maximize else -1
        self.float = set_float_precision(float_precision)
        self.device = device

    @abstractmethod
    def fit(self, train_data: Union[torch.Tensor, DataLoader]) -> None:
        # fit the surrogate model: self.model
        pass

    @abstractmethod
    def get_predictions(self, states: torch.Tensor) -> torch.Tensor:
        pass
