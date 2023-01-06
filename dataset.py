from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd


class Data(Dataset):
    def __init__(self, X_data, y_data, fid):
        self.X_data = X_data
        self.y_data = y_data
        self.fid = fid

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index], self.fid[index]

    def __len__(self):
        return len(self.X_data)


class DataHandler:
    """
    Intialises the train data using env-specific train function
    Scores data using oracle
    """

    def __init__(
        self,
        env,
        normalise_data,
        shuffle_data,
        train_fraction,
        batch_size,
        n_samples,
        seed_data,
        save_data,
        load_data,
        data_path,
    ):
        self.env = env
        self.normalise_data = normalise_data
        self.shuffle_data = shuffle_data
        self.train_fraction = train_fraction
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.seed_data = seed_data
        self.save_data = save_data
        self.load_data = load_data
        self.data_path = data_path

        self.initialise_dataset()

    def initialise_dataset(self):
        """
        - calls env-specific function get a pandas dataframe of samples (list) and energies (list)
        - saves the un-transformed (no proxy transformation) de-normalised data
        - calls preprocess_for_dataloader
        """
        if self.load_data:
            self.dataset = pd.read_csv(self.data_path)
        else:
            # env.make_train_set requires fidelity to do score the samples
            # assume that the fidelity is saved to the dataset as well
            self.dataset = self.env.make_train_set(ntrain=self.n_samples)
        if self.save_data:
            self.dataset.to_csv(self.data_path)
        self.preprocess_for_dataloader()

    def preprocess_for_dataloader(self):
        """
        - converts samples to proxy space
        - normalises the energies
        - shuffles the data
        - splits the data into train and test
        """
        self.samples = torch.FloatTensor(
            list(map(self.env.state2proxy, self.dataset["samples"]))
        )
        self.targets = torch.FloatTensor(self.dataset["energies"])
        self.fidelity = torch.FloatTensor(self.dataset["fidelity"])

        if self.normalise_data:
            self.targets = self.normalise_dataset()
        if self.shuffle_data:
            self.reshuffle()

        # total number of samples is updated with each AL iteration so fraction is multiplied by number of samples at the current iteration
        train_size = int(self.train_fraction * self.samples.shape[0])

        self.train_data = Data(
            self.samples[:train_size],
            self.targets[:train_size],
            self.fidelity[:train_size],
        )
        self.test_data = Data(
            self.samples[train_size:],
            self.targets[train_size:],
            self.fidelity[train_size:],
        )

    def get_statistics(self):
        """
        called each time the dataset is updated so has the most recent metrics
        """
        self.mean = torch.mean(self.targets)
        self.std = torch.std(self.targets)
        return self.mean, self.std

    def normalise_dataset(self, y=None, mean=None, std=None):
        """
        Args:
            y: targets to normalise (tensor)
            mean: mean of targets (tensor)
            std: std of targets (tensor)
        Returns:
            y: normalised targets (tensor)
        """
        if y == None:
            y = self.targets
        if mean == None or std == None:
            mean, std = self.get_statistics()
        y = (y - mean) / std
        return y

    def update_dataset(self, queries, energies):
        """
        Args:
            queries: list of queries [[0, 0], [1, 1], ...]
            energies: list of energies [-0.6, -0.1, ...]
        Update the dataset with new data after AL iteration
        Updates the dataset stats
        Saves the updated dataset if save_data=True
        """
        if self.save_data:
            # loads the saved dataset
            self.dataset = pd.read_csv(self.path_data)
        query_dataset = pd.DataFrame({"samples": queries, "energies": energies})
        self.dataset = pd.concat([self.dataset, query_dataset], ignore_index=True)
        self.preprocess_for_dataloader()
        if self.save_data:
            self.dataset.to_csv(self.path_data)

    def reshuffle(self):
        """
        Reshuffle the entire dataset (called before creating train and test subsets)
        """
        self.samples, self.targets, self.fidelity = shuffle(
            self.samples.numpy(),
            self.targets.numpy(),
            self.fidelity.numpy(),
            random_state=self.seed_data,
        )

    def collate_batch(self, batch):
        """
        Pads till maximum length in the batch
        """
        y, x, = (
            [],
            [],
        )
        for (_sequence, _label) in batch:
            y.append(_label)
            x.append(torch.tensor(_sequence))
        y = torch.tensor(y, dtype=torch.float)
        xPadded = pad_sequence(x, batch_first=True, padding_value=0.0)
        return xPadded, y

    def get_dataloader(self):
        """
        Build and return the dataloader for the networks
        The dataloader should return x and y such that:
            x: self.env.state2proxy(input)
            y: normalised (if need be) energies
        """
        # TODO: Add parameter to config for batch based shuffling (while creating the dataloader) if need be
        tr = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=self.collate_batch,
        )

        te = DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=self.collate_batch,
        )

        return tr, te
