# TOD: fix gfn import
from gflownet.envs.base import make_train_set, make_test_set
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence


class Data(Dataset):
    def __init__(self, X_data, y_data, fid_data):
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

    def __init__(self, config, env, oracle):
        self.config = config
        self.env = env
        self.oracle = oracle

    def initialise_dataset(self):
        """
        - calls env-specific function to initialise sequence
        - scores sequence with oracle
        - normalises/shuffles data (if desired).
        - splits into train and test data.
        """
        x = self.env.make_train_set()
        self.samples = self.env.state2proxy(x)
        self.targets = self.oracle(self.env.state2oracle(x))
        if self.config.normalise_data:
            self.targets = self.normalise_dataset()
        if self.config.shuffle_data:
            self.reshuffle()

        train_size = int(self.config.train_fraction * len(self.samples))

        # TODO: check if samples and targets are lists or tensors
        self.train_dataset = Dataset(
            self.samples[:train_size], self.targets[:train_size]
        )
        self.test_dataset = Dataset(
            self.samples[train_size:], self.targets[train_size:]
        )

    def get_statistics(self):
        self.mean = torch.mean(self.targets)
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
            self.samples, self.targets, random_state=self.seed_data
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
            y: normalised energies
        """

        tr = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            collate_fn=self.collate_batch,
        )

        te = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=self.collate_batch,
        )

        return tr, te
