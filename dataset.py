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
        float_precision,
        n_samples=None,
        fidelity=None,
        rescale=None,
    ):
        self.env = env
        self.normalise_data = normalise_data
        self.train_fraction = train_fraction
        self.n_samples = n_samples
        self.dataloader = dataloader
        self.split = split
        self.path = path
        self.logger = logger
        self.fidelity = fidelity
        self.progress = self.logger.progress
        self.oracle = oracle
        self.logger.set_data_path(self.path.dataset)
        self.dataset_size = dataset_size
        self.device = device
        if hasattr(env, "n_fid"):
            self.n_fid = env.n_fid
            self.sfenv = env.env
        else:
            self.n_fid = 1
            self.sfenv = env
        self.float = set_float_precision(float_precision)
        self.rescale = rescale
        self.initialise_dataset()

    def generate_fidelities(self, states):
        """
        Generates a list of fidelities for the dataset
        """
        n_samples = len(states)
        if self.fidelity.mixed:
            fidelities = torch.randint(low=0, high=self.n_fid, size=(n_samples, 1)).to(
                self.float
            )
            # for i in range(self.n_fid):
            # fidelities[fidelities[:, 0] == i, 0] = self.env.oracle[i].fid
        else:
            # One for each sample
            fidelities = torch.zeros((n_samples * self.n_fid, 1)).to(self.device)
            for i in range(self.n_fid):
                fidelities[i * n_samples : (i + 1) * n_samples, 0] = i
                # self.env.oracle[
                # i
                # ].fid
            states = [states for _ in range(self.n_fid)]
            # TODO: return tensor states here instead of list
        return states, fidelities

    def initialise_dataset(self):
        # TODO: Modify to ensure validation set has equal number of points across fidelities
        """
        Loads the dataset as a dictionary
        OR
        Initialises the dataset using env-specific make_train_set function (returns a dataframe that is converted to a dictionary)

        - dataset['samples']: list of arrays
        - dataset['energies']: list of float values

        If the dataset was initalised and save_data = True, the un-transformed (no proxy transformation) de-normalised data is saved as npy
        """
        if self.path.oracle_dataset:
            # when one dataset without fidelity is given
            # load dataset and convert to states
            if self.path.oracle_dataset.train is not None:
                train = pd.read_csv(self.path.oracle_dataset.train.path)
                train_states = train["samples"].values.tolist()
                if self.path.oracle_dataset.train.get_scores:
                    train_scores = []
                else:
                    train_scores = train["energies"].values.tolist()
            if self.path.oracle_dataset.test is not None:
                test = pd.read_csv(self.path.oracle_dataset.test.path)
                test_states = test["samples"].values.tolist()
                if self.path.oracle_dataset.train.get_scores:
                    test_scores = []
                else:
                    test_scores = test["energies"].values.tolist()
            else:
                test_states = []
                test_scores = []
            states = train_states + test_states
            scores = train_scores + test_scores
            if scores == []:
                scores = None
            states = [
                torch.tensor(self.sfenv.readable2state(sample)) for sample in states
            ]
            # AMP readable2state returns a list of tensors and can take a batch
            # So maybe it would be worthwhile to use that directly
            if hasattr(self.sfenv, "do_state_padding") and self.sfenv.do_state_padding:
                states = pad_sequence(
                    states,
                    batch_first=True,
                    padding_value=self.sfenv.invalid_state_element,
                )
            else:
                states = torch.stack(states)
        else:
            # for AMP this is the implementation
            # dataset = self.env.load_dataset()
            # for grid, I call uniform states. Need to make it uniform
            if self.progress:
                print("Creating dataset of size: ", self.n_samples)
            if self.n_samples is not None:
                states = (
                    torch.tensor(
                        self.sfenv.get_uniform_terminating_states(self.n_samples)
                    ).to(self.device)
                    # .to(self.float)
                )
            else:
                raise ValueError(
                    "Train Dataset size is not provided. n_samples is None"
                )
            scores = None

        if scores is not None:
            scores = torch.tensor(scores, dtype=self.float, device=self.device)
        if self.n_fid > 1 and self.fidelity.do == True:
            states, fidelities = self.generate_fidelities(states)
            fidelities = fidelities.to(states.device)
            states = torch.cat([states, fidelities], dim=1)  # .long()
            state_oracle, fid = self.env.statetorch2oracle(states)
            if scores is None:
                scores = self.env.call_oracle_per_fidelity(state_oracle, fid)
            # Grid
            if hasattr(self.sfenv, "plot_samples_frequency"):
                fig = self.sfenv.plot_samples_frequency(
                    states, title="Train Dataset", rescale=self.rescale
                )
                self.logger.log_figure("train_dataset", fig, use_context=True)
        # TODO: add clause for when n_fid> 1 but fidelity.do=False
        elif self.n_fid == 1 and scores is None:
            state_oracle = self.env.statetorch2oracle(states)
            scores = self.env.oracle(state_oracle)

        if hasattr(self.sfenv, "plot_reward_distribution"):
            fig = self.sfenv.plot_reward_distribution(scores=scores, title="Dataset")
            self.logger.log_figure("initial_dataset", fig, use_context=True)

        if self.split == "random":
            if (
                self.path.oracle_dataset is not None
                and self.path.oracle_dataset.train is not None
            ):
                index = torch.randperm(len(states))
                train_index = index[: int(len(states) * self.train_fraction)]
                test_index = index[int(len(states) * self.train_fraction) :]
                train_states = states[train_index]
                test_states = states[test_index]
                if scores is not None:
                    train_scores = scores[train_index]
                    test_scores = scores[test_index]
                # TODO: can we change this to dtype = self.float and device = cuda
                train_states = train_states.to(self.device)  # long()
                test_states = test_states.to(self.device)  # .long()
                train_scores = train_scores.to(self.float).to(self.device)
                test_scores = test_scores.to(self.float).to(self.device)

        elif self.split == "all_train":
            train_states = states.to(self.device)
            train_scores = scores.to(self.device)
            test_states = torch.Tensor([])
            test_scores = torch.Tensor([])
            # else:
            # train_samples, test_samples = (
            #     dataset[0],
            #     dataset[1],
            # )
            # train_targets = self.oracle(train_samples)
            # test_targets = self.oracle(test_samples)
        # TODO: make general to sf
        if hasattr(self.sfenv, "statetorch2readable"):
            readable_train_samples = [
                self.env.statetorch2readable(sample) for sample in train_states
            ]
            readable_train_dataset = {
                "samples": readable_train_samples,
                "energies": train_scores.tolist(),
            }
        else:
            readable_train_samples = [
                self.env.state2readable(sample) for sample in train_states
            ]
            readable_train_dataset = {
                "samples": readable_train_samples,
                "energies": train_scores.tolist(),
            }
        # Save the raw (un-normalised) dataset
        self.logger.save_dataset(readable_train_dataset, "train")
        self.train_dataset = {"samples": train_states, "energies": train_scores}

        self.train_dataset, self.train_stats = self.preprocess(self.train_dataset)
        self.train_data = Data(
            self.train_dataset["samples"], self.train_dataset["energies"]
        )

        if len(test_states) > 0:
            if hasattr(self.sfenv, "statetorch2readable"):
                readable_test_samples = [
                    self.env.statetorch2readable(sample) for sample in test_states
                ]
                readable_test_dataset = {
                    "samples": readable_test_samples,
                    "energies": test_scores.tolist(),
                }
            else:
                readable_test_samples = [
                    self.env.state2readable(sample) for sample in test_states
                ]
                readable_test_dataset = {
                    "samples": readable_test_samples,
                    "energies": test_scores.tolist(),
                }
            self.logger.save_dataset(readable_test_dataset, "test")
            self.test_dataset = {"samples": test_states, "energies": test_scores}

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
        scores = dataset["energies"]
        # Following is not needed in AMP. Not sure where it is needed
        # if self.n_fid == 1 and self.path.oracle_dataset:
        # state_batch = [self.env.readable2state(sample) for sample in samples]
        # else:
        state_batch = samples
        state_proxy = self.env.statetorch2proxy(state_batch)
        # for when oracle is proxy and grid setup when oracle state is tensor
        if isinstance(state_proxy, tuple):
            state_proxy = torch.concat((state_proxy[0], state_proxy[1]), dim=1)
        if isinstance(state_proxy, list):
            samples = torch.tensor(state_proxy, device=self.device)
        else:
            samples = state_proxy

        dataset = {"samples": samples, "energies": scores}

        stats = self.get_statistics(scores)
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

    def update_dataset(self, states, energies, fidelity=None):
        """
        Args:
            queries: list of queries [[0, 0], [1, 1], ...]
            energies: list of energies [-0.6, -0.1, ...]
        Update the dataset with new data after AL iteration
        Updates the dataset stats
        Saves the updated dataset if save_data=True
        """
        readable_dataset = {
            "samples": [self.env.state2readable(state) for state in states],
            "energies": energies,
        }
        # readable_dataset = readable_dataset.sort_values(by=["energies"])
        self.logger.save_dataset(readable_dataset, "sampled")

        # for grid
        if hasattr(self.sfenv, "plot_samples_frequency"):
            fig = self.sfenv.plot_samples_frequency(
                states, title="Sampled Dataset", rescale=self.rescale
            )
            self.logger.log_figure(
                "post_al_iter_sampled_dataset", fig, use_context=True
            )
        # for AMP
        if hasattr(self.sfenv, "plot_reward_distribution"):
            fig = self.sfenv.plot_reward_distribution(
                scores=energies, title="Post AL Iteration Sampled Dataset"
            )
            self.logger.log_figure(
                "post_al_iter_sampled_dataset", fig, use_context=True
            )

        states = self.env.statebatch2proxy(states)
        if isinstance(states, TensorType) == False:
            states = torch.tensor(
                np.array(states), device=self.device
            )  # dtype=self.float,
        else:
            states = states.to(self.device)  # dtype=self.float
        energies = torch.tensor(energies, dtype=self.float, device=self.device)

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
        y = torch.tensor(y, dtype=self.float, device=self.device)
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
            # num_workers=0,
            # pin_memory=True,
            collate_fn=self.collate_batch,
        )

        test_loader = DataLoader(
            self.test_data,
            batch_size=self.dataloader.test.batch_size,
            shuffle=self.dataloader.test.shuffle,
            # num_workers=0,
            # pin_memory=True,
            collate_fn=self.collate_batch,
        )

        return train_loader, test_loader
