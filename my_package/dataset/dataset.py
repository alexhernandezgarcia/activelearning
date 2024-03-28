from torch.utils.data import Dataset
import torch
from gflownet.utils.common import set_float_precision
from abc import ABC, abstractmethod


class Data(Dataset):
    def __init__(self, X_data, y_data=None, float=torch.float64):
        self.float = float
        self.X_data = X_data.to(self.float)
        self.y_data = None
        if y_data is not None:
            self.y_data = y_data.to(self.float)

    def __getitem__(self, index):
        x_set, y_set = self.preprocess(self.X_data, self.y_data)
        if y_set is not None:
            y_set = y_set[index]

        return x_set[index], y_set

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

    def __init__(self, float_precision=64):
        self.float = set_float_precision(float_precision)

    """
    return dataset loader. a dataset with all current entries is returned in form of a pytorch dataloader.
    """

    @abstractmethod
    def get_dataloader(self):
        pass

    """
    saves/appends future results. the results created by oracles should be saved and appended to the dataset.
    """

    @abstractmethod
    def update_dataset(self):
        pass

    """
    returns a set of candidate data instances
    """

    @abstractmethod
    def get_candidate_set(self):
        pass
