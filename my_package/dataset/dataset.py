from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from gflownet.utils.common import set_device, set_float_precision
from torch.nn.utils.rnn import pad_sequence
from torchtyping import TensorType
from typing import List
from abc import ABC, abstractmethod
# from utils.common import get_figure_plots


class Data(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data.detach().cpu()
        self.y_data = y_data.detach().cpu()

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class Branin_Data(Data):
    """
    Implements the "Data" class for Branin data. 
    
        X_data: array(N, 2) in the domain [0; grid_size]; states (i.e., grid positions) 
        y_data: array(N, 1); scores at each position
        normalize_scores: bool; maps the scores to [0; 1]
        grid_size: int; specifies the width and height of the Branin grid (grid_size x grid_size); used to normalize the state positions
        device: pytorch device used for calculations


    """
    def __init__(self, X_data, y_data, normalize_scores=True, grid_size=100, device="cpu"):
        super().__init__(X_data, y_data)
        self.normalize_scores = normalize_scores
        self.device = device
        self.grid_size = grid_size
        self.stats = self.get_statistics(y_data)
        

    def get_statistics(self, y):
        """
        called each time the dataset is updated so has the most recent metrics
        """
        dict = {}
        dict["mean"] = torch.mean(y)
        dict["std"] = torch.std(y)
        dict["max"] = torch.max(y)
        dict["min"] = torch.min(y)
        return dict


    def __getitem__(self, index):
        X = self.X_data[index]
        y = self.y_data[index]
        return self.preprocess(X, y)
    
    def normalize(self, y):
        """
        Args:
            y: targets to normalize (tensor)
        Returns:
            y: normalized targets (tensor)
        """
        y = (y - self.stats["min"]) / (self.stats["max"] - self.stats["min"])
        # y = (y - stats["mean"]) / stats["std"]
        return y

    def denormalize(self, y):
        """
        Args:
            y: targets to denormalize (tensor)
        Returns:
            y: denormalized targets (tensor)
        """
        y = y * (self.stats["max"] - self.stats["min"]) + self.stats["min"]
        # y = y * stats["std"] + stats["mean"]
        return y

    
    def preprocess(self, X, y):
        """
        - normalizes the scoers
        - normalizes the states
        """
        states = X / self.grid_size
        if self.normalize_scores:
            scores = self.normalize(y)

        return states, scores



class AL_DatasetHandler(ABC):
    """
    loads initial dataset. this contains the parameters (X), the target (y), and for multiple oracles, the oracle ID that created this datapoint. 
    """
    @abstractmethod 
    def __init__(self):
        pass

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


class Branin_DatasetHandler(AL_DatasetHandler):
    """
    loads initial dataset. a path must be passed from where the initial csv file is loaded. it also checks whether other iteration steps are present in the directory and loads them too.

    return dataset. the dataloader contains a preprocessing function that normalizes the data according to the Branin Grid environment.

    saves/appends future results. appends the oracle data to the dataset that is loaded in memory. additionally saves the results from the current iteration in a new csv file.
    """

    def __init__(
        self,
        normalize_data=True,
        train_fraction=0.8,
        batch_size=256,
        shuffle=True,
        train_path="storage/branin/sf/data_train.csv",
        test_path=None,
        target_factor=1.0,
        device="cpu",
        float_precision=64,
        grid_size=100,
    ):
        self.normalize_data = normalize_data
        self.train_fraction = train_fraction
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_path = train_path
        self.test_path = test_path
        self.target_factor = target_factor
        self.device = device
        self.float = set_float_precision(float_precision)
        self.grid_size = grid_size

        self.initialise_dataset()


    def scale_by_target_factor(self, data):
        if data is not None:
            data = torch.tensor(data)
            data = data * self.target_factor
            indices = torch.where(data == -0.0)
            data[indices] = 0.0
        return data
    
    def statetorch2readable(self, state, alphabet={}):
        """
        Dataset Handler in activelearning deals only in tensors. This function converts the tesnor to readble format to save the train dataset
        """
        assert torch.eq(state.to(torch.long), state).all()
        state = state.to(torch.long)
        state = state.detach().cpu().numpy()
        return str(state).replace("(", "[").replace(")", "]").replace(",", "")

    def readable2state(self, readable):
        """
        Converts a human-readable string representing a state into a state as a list of
        positions.
        """
        return [int(el) for el in readable.strip("[]").split(" ") if el != ""]

    def state2readable(self, state = None):
        """
        Converts a state (a list of positions) into a human-readable string
        representing a state.
        """
        state = self._get_state(state)
        return str(state).replace("(", "[").replace(")", "]").replace(",", "")

    def initialise_dataset(self):
        """
        Loads the initial dataset from a directory
        """
        train_scores = []
        test_scores = []
        train_samples = []
        test_samples = []

        # TODO: load all iterations that are saved in the directory
        # load train data
        train_states = torch.Tensor()
        if self.train_path is not None:
            train = pd.read_csv(self.train_path)
            train_samples = train["samples"].values.tolist()
            train_scores = train["energies"].values.tolist()
            train_states = torch.stack([
                torch.tensor(self.readable2state(sample)) for sample in train_samples
            ])


        # load test data
        test_states = torch.Tensor()
        if self.test_path is not None:
            test = pd.read_csv(self.test_path)
            test_samples = test["samples"].values.tolist()
            test_scores = test["energies"].values.tolist()
            test_states = torch.stack([
                torch.tensor(self.readable2state(sample)) for sample in test_samples
            ])

        # TODO: check if we need to scale; if yes, we might want to put this in the Data preprocessing
        train_scores = self.scale_by_target_factor(train_scores)
        test_scores = self.scale_by_target_factor(test_scores)

        # if we don't have test data and we specified a train_fraction, 
        # use a random subsample from the train data as test data
        if len(test_states) <= 0 and self.train_fraction < 1.0:
            index = torch.randperm(len(train_states))
            train_index = index[: int(len(train_states) * self.train_fraction)]
            test_index = index[int(len(train_states) * self.train_fraction) :]
            test_states = train_states[test_index]
            train_states = train_states[train_index]
            test_scores = train_scores[test_index]
            train_scores = train_scores[train_index]

        # send to device
        # train_states = train_states.to(self.device)
        # test_states = test_states.to(self.device)
        # train_scores = train_scores.to(self.float).to(self.device)
        # test_scores = test_scores.to(self.float).to(self.device)

        self.train_data = Branin_Data(
            train_states, 
            train_scores,
            normalize_data=self.normalize_data,
            float=self.float,
            device=self.device
        )

        if len(test_states) > 0:
            self.test_data = Branin_Data(
                self.test_dataset["states"], 
                self.test_dataset["energies"],
                normalize_data=self.normalize_data,
                float=self.float,
                device=self.device
            )
        else:
            self.test_data = None


    def collate_batch(self, batch):
        """
        Pads till maximum length in the batch
        """
        y, x = (
            [],
            [],
        )
        for (_sequence, _label) in batch:
            y.append(_label)
            x.append(_sequence)
        y = torch.tensor(y, dtype=self.float)  # , device=self.device
        xPadded = pad_sequence(x, batch_first=True, padding_value=0.0)
        return xPadded, y

    def get_dataloader(self):
        """
        Build and return the dataloader for the networks
        The dataloader should return x and y such that:
            x: self.statebatch2proxy(input)
            y: normalized (if need be) energies
        """
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=2,
            pin_memory=True,
            collate_fn=self.collate_batch,
        )
        test_loader = None
        if self.test_data is not None:
            test_loader = DataLoader(
                self.test_data,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=2,
                pin_memory=True,
                collate_fn=self.collate_batch,
            )

        return train_loader, test_loader


    def convert_to_tensor(self, data):
        """
        Converts a list of arrays to a tensor
        """
        if isinstance(data, TensorType):
            return data
        elif isinstance(data, List) and len(data) == 0:
            return torch.tensor(data, dtype=self.float, device=self.device)
        elif isinstance(data, List) and isinstance(data[0], TensorType):
            return torch.stack(data).to(self.device)
        elif isinstance(data, List):
            return torch.tensor(data, dtype=self.float, device=self.device)
        else:
            raise NotImplementedError(
                "Data type not recognized for conversion to tensor"
            )

    def update_dataset(self, states, energies, fidelity=None):
        # TODO christina: not refactored yet!
        """
        Args:
            queries: list of queries [[0, 0], [1, 1], ...]
            energies: list of energies [-0.6, -0.1, ...]
        Update the dataset with new data after AL iteration
        Updates the dataset stats
        Saves the updated dataset if save_data=True
        """

        energies = torch.tensor(energies, dtype=self.float, device=self.device)

        samples = [self.state2readable(state) for state in states]
        readable_dataset = {
            "samples": samples,
            "energies": energies.tolist(),
        }
        energies = self.scale_by_target_factor(energies)
        states_proxy = self.statebatch2proxy(states)

        train_energies, test_energies = [], []
        train_states, test_states = [], []
        train_samples, test_samples = [], []
        train_states_proxy, test_states_proxy = [], []
        for sample, state, state_proxy, energy in zip(
            samples, states, states_proxy, energies
        ):
            if np.random.uniform() < (1 / 10):
                test_samples.append(sample)
                test_states.append(state)
                test_states_proxy.append(state_proxy)
                test_energies.append(energy.item())
            else:
                train_samples.append(sample)
                train_states.append(state)
                train_states_proxy.append(state_proxy)
                train_energies.append(energy.item())

        test_states_proxy = self.convert_to_tensor(test_states_proxy)
        train_states_proxy = self.convert_to_tensor(train_states_proxy)
        test_energies = self.convert_to_tensor(test_energies)
        train_energies = self.convert_to_tensor(train_energies)

        if self.normalize_data:
            self.train_dataset["energies"] = self.denormalize(
                self.train_dataset["energies"], stats=self.train_stats
            )
            if self.test_dataset is not None:
                self.test_dataset["energies"] = self.denormalize(
                    self.test_dataset["energies"], stats=self.test_stats
                )

        self.train_dataset["energies"] = torch.cat(
            (self.train_dataset["energies"], train_energies), dim=0
        )
        if self.test_dataset is not None:
            self.test_dataset["energies"] = torch.cat(
                (self.test_dataset["energies"], test_energies), dim=0
            )

        self.train_dataset["states"] = torch.cat(
            (self.train_dataset["states"], train_states_proxy), dim=0
        )
        if self.test_dataset is not None:
            self.test_dataset["states"] = torch.cat(
                (self.test_dataset["states"], test_states_proxy), dim=0
            )

        self.train_stats = self.get_statistics(self.train_dataset["energies"])
        if self.test_dataset is not None:
            self.test_stats = self.get_statistics(self.test_dataset["energies"])
        if self.normalize_data:
            self.train_dataset["energies"] = self.normalize(
                self.train_dataset["energies"], self.train_stats
            )
            if self.test_dataset is not None:
                self.test_dataset["energies"] = self.normalize(
                    self.test_dataset["energies"], self.test_stats
                )
        self.train_data = Data(
            self.train_dataset["states"], self.train_dataset["energies"]
        )
        if self.test_dataset is not None:
            self.test_data = Data(
                self.test_dataset["states"], self.test_dataset["energies"]
            )


