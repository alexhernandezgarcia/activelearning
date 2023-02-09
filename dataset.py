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


class Data(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


# TODO: Rename samples to states
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
        path,
        logger,
        oracle,
        split,
        dataset_size,
        device,
        n_samples,
        mixed_fidelity,
        float_precision,
    ):
        self.env = env
        self.normalise_data = normalise_data
        self.train_fraction = train_fraction
        self.n_samples = n_samples
        self.dataloader = dataloader
        self.split = split
        self.path = path
        self.logger = logger
        self.mixed_fidelity = mixed_fidelity
        self.progress = self.logger.progress
        self.oracle = oracle
        self.logger.set_data_path(self.path.dataset)
        self.dataset_size = dataset_size
        self.device = device
        self.n_fid = self.env.n_fid
        self.float = set_float_precision(float_precision)
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
            train = pd.read_csv(self.path.oracle_dataset.train)
            test = pd.read_csv(self.path.oracle_dataset.test)
            train_samples = train["samples"].values.tolist()
            train_targets = train["energies"].values.tolist()
            test_samples = test["samples"].values.tolist()
            test_targets = test["energies"].values.tolist()

        else:
            # for AMP this is the implementation
            # dataset = self.env.load_dataset()
            # for grid, I call uniform states. Need to make it uniform
            if self.progress:
                print("Creating dataset of size: ", self.n_samples)
            states = (
                torch.Tensor(
                    self.env.env.get_uniform_terminating_states(self.n_samples)
                )
                .to(self.device)
                .long()
            )
            if self.mixed_fidelity == True:
                fidelities = torch.randint(0, self.n_fid, (len(states), 1)).to(
                    self.device
                )
            else:
                fidelities = torch.zeros((len(states) * self.n_fid, 1)).to(self.device)
                for i in range(self.n_fid):
                    fidelities[i * len(states) : (i + 1) * len(states), 0] = i
                states = states.repeat(self.n_fid, 1)

            state_fid = torch.cat([states, fidelities], dim=1).long()
            states_fid_oracle = self.env.statetorch2oracle(state_fid)
            scores = self.env.call_oracle_per_fidelity(states_fid_oracle)

            if hasattr(self.env.env, "plot_samples_frequency"):
                fig = self.env.env.plot_samples_frequency(states, title="Train Dataset")
                self.logger.log_figure("train_dataset", fig, use_context=True)

            # fig = self.env.plot_reward_samples(states, scores, "Train Dataset")
            # self.logger.log_figure(fig, "train_dataset")
            # index = states.long().detach().cpu().numpy()
            # # data_scores = np.reshape(scores.detach().cpu().numpy(), (10, 10))
            # grid_scores = np.ones((20, 20)) * (5.0)
            # grid_scores[index[:, 0], index[:, 1]] = scores.detach().cpu().numpy()
            # plt.imshow(grid_scores)
            # plt.colorbar()
            # plt.title("Train Data")
            # plt.savefig(
            #     "/home/mila/n/nikita.saxena/activelearning/storage/grid/train_data.png"
            # )
            # plt.close()
            samples = state_fid.tolist()
            targets = scores.tolist()

        if self.split == "random":
            # randomly select 10 element from the list train_samples and test_samples
            if (
                self.path.oracle_dataset is not None
                and self.path.oracle_dataset.train is not None
            ):
                samples = train_samples + test_samples
                targets = train_targets + test_targets
            train_samples, test_samples, train_targets, test_targets = train_test_split(
                samples, targets, train_size=self.train_fraction
            )
        elif self.split == "all_train":
            train_samples = samples
            train_targets = targets
            test_samples = []
            test_targets = []
            # else:
            # train_samples, test_samples = (
            #     dataset[0],
            #     dataset[1],
            # )
            # train_targets = self.oracle(train_samples)
            # test_targets = self.oracle(test_samples)

        readable_train_samples = [
            self.env.state2readable(sample) for sample in train_samples
        ]
        readable_train_dataset = {
            "samples": readable_train_samples,
            "energies": train_targets,
        }
        # Save the raw (un-normalised) dataset
        self.logger.save_dataset(readable_train_dataset, "train")
        self.train_dataset = {"samples": train_samples, "energies": train_targets}

        self.train_dataset, self.train_stats = self.preprocess(self.train_dataset)
        self.train_data = Data(
            self.train_dataset["samples"], self.train_dataset["energies"]
        )

        if len(test_samples) > 0:
            readable_test_samples = [
                self.env.state2readable(sample) for sample in test_samples
            ]
            readable_test_dataset = {
                "samples": readable_test_samples,
                "energies": test_targets,
            }
            self.logger.save_dataset(readable_test_dataset, "test")
            self.test_dataset = {"samples": test_samples, "energies": test_targets}

            self.test_dataset, self.test_stats = self.preprocess(self.test_dataset)
            self.test_data = Data(
                self.test_dataset["samples"], self.test_dataset["energies"]
            )
        else:
            self.test_dataset = None
            self.test_data = None
            self.test_stats = None

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
            if self.test_stats is not None:
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
        if self.path.oracle_dataset:
            state_batch = [self.env.readable2state(sample) for sample in samples]
        else:
            state_batch = samples
        state_proxy = self.env.statebatch2proxy(state_batch)
        if isinstance(state_proxy, list):
            samples = torch.FloatTensor(state_proxy)
        else:
            samples = state_proxy
        # TODO: delete this keep everything in tensor only, Remove list conversion
        targets = torch.tensor(targets, dtype=self.float)

        dataset = {"samples": samples, "energies": targets}

        stats = self.get_statistics(targets)
        if self.normalise_data:
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

    def update_dataset(self, states, energies):
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
        readable_dataset = {
            "samples": [self.env.state2readable(state) for state in states],
            "energies": energies,
        }
        self.logger.save_dataset(readable_dataset, "sampled")

        # plot the frequency of sampled dataset
        if hasattr(self.env.env, "plot_samples_frequency"):
            fig = self.env.env.plot_samples_frequency(states, title="Sampled Dataset")
            self.logger.log_figure(
                "post_al_iter_sampled_dataset", fig, use_context=True
            )

        states = self.env.statebatch2proxy(states)
        if isinstance(states, list):
            states = torch.FloatTensor(np.array(states))
        energies = torch.tensor(energies, dtype=self.float)

        if self.normalise_data:
            self.train_dataset["energies"] = self.denormalise(
                self.train_dataset["energies"], stats=self.train_stats
            )

        self.train_dataset["energies"] = torch.cat(
            (self.train_dataset["energies"], energies), dim=0
        )
        self.train_dataset["samples"] = torch.cat(
            (self.train_dataset["samples"], states), dim=0
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
            print(prefix + "Updated Dataset Statistics")
            print(
                "Train \n \t Mean Score:{:.2f} \n \t  Std:{:.2f} \n \t Min Score:{:.2f} \n \t Max Score:{:.2f}".format(
                    self.train_stats["mean"],
                    self.train_stats["std"],
                    self.train_stats["min"],
                    self.train_stats["max"],
                )
            )
            if self.test_stats is not None:
                print(
                    "Test \n \t Mean Score:{:.2f}  \n \t Std:{:.2f} \n \t Min Score:{:.2f} \n \t Max Score:{:.2f}".format(
                        self.test_stats["mean"],
                        self.test_stats["std"],
                        self.test_stats["min"],
                        self.test_stats["max"],
                    )
                )

        # Update data_train.csv
        path = self.logger.data_path.parent / Path("data_train.csv")
        dataset = pd.read_csv(path, index_col=0)
        dataset = pd.concat([dataset, pd.DataFrame(readable_dataset)])
        self.logger.save_dataset(dataset, "train")

    def reshuffle(self):
        # TODO: Deprecated. Remove once sure it's not used.
        """
        Reshuffle the entire dataset (called before creating train and test subsets)
        """
        self.samples, self.targets = shuffle(
            self.samples.numpy(),
            self.targets.numpy(),
        )

    def collate_batch(self, batch):
        """
        Pads till maximum length in the batch
        """
        y, x, fid = (
            [],
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
            x: self.env.statebatch2proxy(input)
            y: normalised (if need be) energies
        """
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.dataloader.train.batch_size,
            shuffle=self.dataloader.train.shuffle,
            num_workers=0,
            pin_memory=False,
            collate_fn=self.collate_batch,
        )

        test_loader = DataLoader(
            self.test_data,
            batch_size=self.dataloader.test.batch_size,
            shuffle=self.dataloader.test.shuffle,
            num_workers=0,
            pin_memory=False,
            collate_fn=self.collate_batch,
        )

        return train_loader, test_loader
