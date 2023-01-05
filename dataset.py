# from gflownet.src.gflownet.envs.base import make_train_set, make_test_set
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class Data(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

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
    ):
        self.env = env
        self.normalise_data = normalise_data
        self.shuffle_data = shuffle_data
        self.train_fraction = train_fraction
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.seed_data = seed_data
        self.initialise_dataset()

    def initialise_dataset(self):
        """
        - calls env-specific function to initialise sequence
        - scores sequence with oracle
        - normalises/shuffles data (if desired).
        - splits into train and test data.
        """
        dataset = self.env.make_train_set(ntrain=self.n_samples)
        self.samples = torch.FloatTensor(
            list(map(self.env.state2proxy, dataset["samples"]))
        )
        self.targets = torch.FloatTensor(dataset["energies"])
        if self.normalise_data:
            self.targets = self.normalise_dataset()
        if self.shuffle_data:
            self.reshuffle()

        train_size = int(self.train_fraction * self.n_samples)

        # TODO: check if samples and targets are lists or tensors
        self.train_dataset = Data(self.samples[:train_size], self.targets[:train_size])
        self.test_dataset = Data(self.samples[train_size:], self.targets[train_size:])

    def get_statistics(self):
        # find mean of elements in list
        self.mean = torch.mean(self.targets)
        # find standard deviation of elements in list
        self.std = torch.std(self.targets)
        return self.mean, self.std

    def normalise_dataset(self, y=None, mean=None, std=None):
        if y == None:
            y = self.targets
        if mean == None or std == None:
            mean, std = self.get_statistics()
        y = (y - mean) / std
        return y

    def update_dataset(self, **kwargs):
        """
        Update the dataset with new data after AL iteration
        Also update the dataset stats
        """
        pass

    def reshuffle(self):
        self.samples, self.targets = shuffle(
            self.samples.numpy(), self.targets.numpy(), random_state=self.seed_data
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

        tr = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=self.collate_batch,
        )

        te = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=self.collate_batch,
        )

        return tr, te
