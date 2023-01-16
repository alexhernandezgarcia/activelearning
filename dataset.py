from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd


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
        train_fraction,
        dataloader,
        n_samples,
        path,
        logger,
        oracle,
        split,
    ):
        self.env = env
        self.normalise_data = normalise_data
        self.train_fraction = train_fraction
        self.dataloader = dataloader
        self.n_samples = n_samples
        self.split = split
        self.path = path
        self.logger = logger
        self.progress = self.logger.progress
        self.oracle = oracle
        self.logger.set_data_path(self.path.dataset)
        self.initialise_dataset()

    def initialise_dataset(self):
        """
        Loads the dataset as a dictionary
        OR
        Initialises the dataset using env-specific make_train_set function (returns a dataframe that is converted to a dictionary)

        - dataset['samples']: list of arrays
        - dataset['energies']: list of float values

        If the dataset was initalised and save_data = True, the un-transformed (no proxy transformation) de-normalised data is saved as npy
        """
        if self.path.oracle_dataset:
            # TODO: Refine logic when I have a dataset
            path = self.logger.data_path.parent / Path(path.dataset)
            dataset = np.load(self.data_path, allow_pickle=True)
            dataset = dataset.item()

        else:
            dataset = self.env.load_dataset()

        if self.split == "random":
            train_samples, test_samples = train_test_split(
                dataset, train_size=self.train_fraction
            )
        else:
            train_samples, train_targets, test_samples, test_targets = (
                dataset[0],
                dataset[1],
                dataset[2],
                dataset[3],
            )
        # UNCOMMENT ONCE DONE TESTING
        # train_targets = self.oracle(train_samples)
        # test_targets = self.oracle(test_samples)

        self.train_dataset = {"samples": train_samples, "energies": train_targets}
        self.test_dataset = {"samples": test_samples, "energies": test_targets}

        # Save the raw (un-normalised) dataset
        self.logger.save_dataset(self.train_dataset, self.test_dataset)

        self.train_dataset, self.train_stats = self.preprocess(self.train_dataset)
        self.train_data = Data(
            self.train_dataset["samples"], self.train_dataset["energies"]
        )

        self.test_dataset, self.test_stats = self.preprocess(self.test_dataset)
        self.test_data = Data(
            self.test_dataset["samples"], self.test_dataset["energies"]
        )

        # Log the dataset statistics
        self.logger.log_dataset_stats(self.train_stats, self.test_stats)
        if self.progress:
            prefix = "Normalised " if self.normalise_data else ""
            print(prefix + "Dataset Statistics")
            print(
                "Train Data \n \t Mean Score:{:.2f} \n \t Std:{:.2f} \n \t Min Score:{:.2f} \n \t Max Score:{:.2f}".format(
                    self.train_stats["mean"],
                    self.train_stats["std"],
                    self.train_stats["min"],
                    self.train_stats["max"],
                )
            )
            print(
                "Test Data \n \t Mean Score:{:.2f}  \n \t Std:{:.2f} \n \t Min Score:{:.2f} \n \t Max Score:{:.2f}".format(
                    self.test_stats["mean"],
                    self.test_stats["std"],
                    self.test_stats["min"],
                    self.test_stats["max"],
                )
            )

    def preprocess(self, dataset):
        """
        - converts samples to proxy space
        - normalises the energies
        - shuffles the data
        - splits the data into train and test
        """
        samples = dataset["samples"]
        targets = dataset["energies"]

        samples = torch.FloatTensor(
            np.array(
                [
                    self.env.state2proxy(self.env.readable2state(sample))
                    for sample in samples
                ]
            )
        )
        targets = torch.FloatTensor(targets)

        dataset = {"samples": samples, "energies": targets}

        stats = self.get_statistics(targets)
        dataset["energies"] = self.normalise(dataset["energies"], stats)

        return dataset, stats

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

    def normalise(self, y, stats):
        """
        Args:
            y: targets to normalise (tensor)
            mean: mean of targets (tensor)
            std: std of targets (tensor)
        Returns:
            y: normalised targets (tensor)
        """
        y = (y - stats["mean"]) / stats["std"]
        return y

    def denormalise(self, y, stats):
        """
        Args:
            y: targets to denormalise (tensor)
            mean: mean of targets (tensor)
            std: std of targets (tensor)
        Returns:
            y: denormalised targets (tensor)
        """
        y = y * stats["std"] + stats["mean"]
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
        # TODO: Deprecate the "if" statement
        # if self.save_data:
        #     # load the saved dataset
        #     dataset = np.load(self.data_path, allow_pickle=True)
        #     self.dataset = dataset.item()
        #     # index_col=False

        queries = torch.FloatTensor(
            np.array([self.env.state2proxy(sample) for sample in queries])
        )
        energies = torch.FloatTensor(energies)

        if self.normalise_data:
            self.train_dataset["energies"] = self.denormalise(
                self.train_dataset["energies"], stats=self.train_stats
            )

        self.train_dataset["energies"] = torch.cat(
            (self.train_dataset["energies"], energies), dim=0
        )
        self.train_dataset["samples"] = torch.cat(
            (self.train_dataset["samples"], queries), dim=0
        )

        self.train_stats = self.get_statistics(self.train_dataset["energies"])
        if self.normalise_data:
            self.train_dataset["energies"] = self.normalise(
                self.train_dataset["energies"], self.train_stats
            )
        self.train_data = Data(
            self.train_dataset["samples"], self.train_dataset["energies"]
        )

        self.logger.log_dataset_stats(self.train_stats, self.test_stats)
        if self.progress:
            prefix = "Normalised " if self.normalise_data else ""
            print(prefix + "Dataset Statistics")
            print(
                "Train \n \t Mean Score:{:.2f} \n \t  Std:{:.2f} \n \t Min Score:{:.2f} \n \t Max Score:{:.2f}".format(
                    self.train_stats["mean"],
                    self.train_stats["std"],
                    self.train_stats["min"],
                    self.train_stats["max"],
                )
            )
            print(
                "Test \n \t Mean Score:{:.2f}  \n \t Std:{:.2f} \n \t Min Score:{:.2f} \n \t Max Score:{:.2f}".format(
                    self.test_stats["mean"],
                    self.test_stats["std"],
                    self.test_stats["min"],
                    self.test_stats["max"],
                )
            )

    def reshuffle(self):
        # TODO: Deprecated. Remove once sure it's not used.
        """
        Reshuffle the entire dataset (called before creating train and test subsets)
        """
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
            x.append(_sequence)
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
            self.train_data,
            batch_size=self.dataloader.train.batch_size,
            shuffle=self.dataloader.train.shuffle,
            num_workers=0,
            pin_memory=False,
            collate_fn=self.collate_batch,
        )

        te = DataLoader(
            self.test_data,
            batch_size=self.dataloader.test.batch_size,
            shuffle=self.dataloader.test.shuffle,
            num_workers=0,
            pin_memory=False,
            collate_fn=self.collate_batch,
        )

        return tr, te
