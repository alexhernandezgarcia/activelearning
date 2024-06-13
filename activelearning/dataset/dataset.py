from torch.utils.data import Dataset, DataLoader
import torch
from gflownet.utils.common import set_float_precision
from gflownet.envs.base import GFlowNetEnv
from abc import ABC, abstractmethod
from typing import Optional, Union, Callable
import numpy as np


class Data(Dataset):
    def __init__(
        self,
        X_data: torch.Tensor,
        y_data: Optional[torch.Tensor] = None,
        state2result: Optional[Callable[[torch.Tensor], any]] = None,
        float: Union[torch.dtype, int] = torch.float64,
    ) -> None:
        self.float = float
        self.X_data = X_data
        self.y_data = None
        if y_data is not None:
            self.y_data = y_data

        self.state2result = state2result
        self.shape = X_data.shape

    def __getitem__(
        self, index: Union[int, slice, list, np.array]
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:

        x_set = self.X_data[index].to(self.float)
        y_set = None
        if self.y_data is not None:
            y_set = self.y_data[index].to(self.float)
        x_set, y_set = self.preprocess(x_set, y_set)

        if y_set is None:
            return x_set
        return x_set, y_set

    def __len__(self):
        return len(self.X_data)

    def get_raw_items(
        self, index: Union[int, slice, list, np.array] = None
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if index is None:
            if self.y_data is None:
                return self.X_data
            return self.X_data, self.y_data
        if self.y_data is None:
            return self.X_data[index]
        return self.X_data[index], self.y_data[index]

    def preprocess(self, X, y):
        return self.state2result(X), y

    def append(self, X, y):
        """
        append new instances to the data
        """
        self.X_data = torch.cat((self.X_data, X), 0)
        if self.y_data is not None:
            self.y_data = torch.cat((self.y_data, y), 0)


class DatasetHandler(ABC):
    """
    loads initial dataset. this contains the parameters (X), the target (y), and for multiple oracles, the oracle ID that created this datapoint.
    """

    def __init__(
        self,
        env: GFlowNetEnv,
        float_precision: int = 64,
        batch_size=256,
        shuffle=True,
    ):
        self.env = env
        self.float = set_float_precision(float_precision)
        self.batch_size = batch_size
        self.shuffle = shuffle

    """
    return the maximum target value
    """

    @abstractmethod
    def maxY(self) -> Union[float, torch.Tensor]:
        pass

    """
    return the minimum target value
    """

    @abstractmethod
    def minY(self) -> Union[float, torch.Tensor]:
        pass

    """
    return dataset loader. a dataset with all current entries is returned in form of a pytorch dataloader.
    """

    @abstractmethod
    def get_dataloader(self) -> tuple[DataLoader, Optional[DataLoader]]:
        pass

    """
    saves/appends future results. the results created by oracles should be saved and appended to the dataset.
    """

    @abstractmethod
    def update_dataset(self, X: torch.Tensor, y: torch.tensor) -> torch.tensor:
        pass

    """
    returns a set of candidate data instances
    """

    @abstractmethod
    def get_candidate_set(self) -> tuple[Union[Data, DataLoader], Optional[any]]:
        pass

    """
    returns a set of sample states as a Data Object
    """

    @abstractmethod
    def get_custom_dataset(self, samples: torch.Tensor) -> Data:
        pass

    """
    transforms states into oracle format. by default the format is the same format as states2proxy
    """

    def states2oracle(self, samples):
        return self.get_custom_dataset(samples)

    def prepare_oracle_dataloader(self, dataset: Data, sample_idcs=None):
        return dataset
