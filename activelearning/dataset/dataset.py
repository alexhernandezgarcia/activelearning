from torch.utils.data import Dataset, DataLoader
import torch
from gflownet.utils.common import set_float_precision
from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np


class Data(Dataset):
    def __init__(
        self,
        X_data: torch.Tensor,
        y_data: Optional[torch.Tensor] = None,
        float: Union[torch.dtype, int] = torch.float64,
    ) -> None:
        self.float = float
        self.X_data = X_data
        self.y_data = None
        if y_data is not None:
            self.y_data = y_data

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

    def preprocess(self, X, y):
        return X, y

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
        float_precision: int = 64,
        batch_size=256,
        shuffle=True,
    ):
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
    prepares a set of sample data instances
    only needed if there is a missmatch between the data used for the surrogate and the data for the oracle
    """

    def prepare_dataset_for_oracle(self, samples, sample_idcs):
        return samples

    def prepare_oracle_dataloader(self, dataset: Data, sample_idcs=None):
        return dataset
