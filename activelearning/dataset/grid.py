from activelearning.dataset.dataset import DatasetHandler, Data
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import torch
from gflownet.envs.grid import Grid as GridEnv
from typing import Optional, Callable


class GridData(Data):
    """
    Implements the "Data" class for Grid data.

        X_data: array(N, d) in the domain [0; grid_size], where d is the dimensionality of the grid world (e.g. for Branin d=2; for Hartmann d=6); states (i.e., grid positions)
        y_data: array(N, 1); scores at each position
        normalize_scores: bool; maps the scores to [0; 1]
        state2result: function that takes raw states (torch.Tensor) (aka environment format) and transforms them into the desired format;
            in case of GFN environments, this can be the states2proxy function

    """

    def __init__(
        self,
        X_data,
        y_data=None,
        state2result: Optional[Callable[[torch.Tensor], any]] = None,
        normalize_scores=True,
        float=torch.float64,
    ):
        super().__init__(
            X_data=X_data, y_data=y_data, state2result=state2result, float=float
        )
        self.normalize_scores = normalize_scores
        self.stats = self.get_statistics(y_data)

    def get_statistics(self, y):
        """
        called each time the dataset is updated so has the most recent metrics
        """
        if y is None:
            return None

        dict = {}
        dict["mean"] = torch.mean(y)
        dict["std"] = torch.std(y)
        dict["max"] = torch.max(y)
        dict["min"] = torch.min(y)
        return dict

    def append(self, X, y):
        """
        append new instances to the data
        """
        super().append(X, y)
        self.stats = self.get_statistics(self.y_data)  # update the score statistics

    def preprocess(self, X, y):
        """
        - normalizes the scores
        - normalizes the states
        """
        states = X
        if self.state2result is not None:
            if len(states.shape) == 1:
                states = self.state2result([states])[0]
            else:
                states = self.state2result(states)
        scores = None
        if self.normalize_scores and y is not None:
            scores = (y - self.stats["min"]) / (self.stats["max"] - self.stats["min"])

        return states, scores


class GridDatasetHandler(DatasetHandler):
    """
    loads initial dataset. a path must be passed from where the initial csv file is loaded. it also checks whether other iteration steps are present in the directory and loads them too.

    return dataset. the dataloader contains a preprocessing function that normalizes the data according to the Grid environment.

    saves/appends future results. appends the oracle data to the dataset that is loaded in memory. additionally saves the results from the current iteration in a new csv file.
    """

    def __init__(
        self,
        env: GridEnv,
        train_fraction=0.8,
        batch_size=256,
        shuffle=True,
        train_path=None,
        test_path=None,
        float_precision=64,
    ):
        super().__init__(
            env=env,
            float_precision=float_precision,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        self.train_fraction = train_fraction
        self.train_path = train_path
        self.test_path = test_path

        self.initialise_dataset()

    # def proxy2state(self, proxy_state):
    #     """
    #     Converts a proxy state into the environment state format.
    #     """
    #     domain_01 = (proxy_state + 1) / 2
    #     return domain_01 * (self.env.length - 1)

    def maxY(self):
        _, train_y = self.train_data[:]
        return train_y.max()

    def minY(self):
        _, train_y = self.train_data[:]
        return train_y.min()

    def initialise_dataset(self):
        """
        Loads the initial dataset from a directory
        """

        # TODO: load all iterations that are saved in the directory
        # load train data
        train_states = torch.Tensor()
        train_scores = torch.Tensor()
        if self.train_path is not None:
            train = pd.read_csv(self.train_path)
            train_samples_X = train["samples"].values.tolist()
            train_samples_y = train["energies"].values.tolist()
            train_scores = torch.tensor(train_samples_y)
            train_states = torch.stack(
                [
                    torch.tensor(self.env.readable2state(sample))
                    for sample in train_samples_X
                ]
            )

        # load test data
        test_states = torch.Tensor()
        test_scores = torch.Tensor()
        if self.test_path is not None:
            test = pd.read_csv(self.test_path)
            test_samples_X = test["samples"].values.tolist()
            test_samples_y = test["energies"].values.tolist()
            test_scores = torch.tensor(test_samples_y)
            test_states = torch.stack(
                [
                    torch.tensor(self.env.readable2state(sample))
                    for sample in test_samples_X
                ]
            )

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

        self.train_data = GridData(
            train_states,
            train_scores,
            float=self.float,
            state2result=self.env.states2proxy,
        )

        if len(test_states) > 0:
            self.test_data = GridData(
                test_states,
                test_scores,
                float=self.float,
                state2result=self.env.states2proxy,
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
        for _sequence, _label in batch:
            y.append(_label)
            x.append(_sequence)
        y = torch.tensor(y, dtype=self.float)
        xPadded = pad_sequence(x, batch_first=True, padding_value=0.0)
        return xPadded, y

    def get_dataloader(self):
        """
        Build and return the dataloader for the networks
        The dataloader should return x and y such that:
            x: in the domain [0; 1]
            y: normalized (if need be) energies [0; 1]
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

    def update_dataset(self, X, y, save_path=None):
        """
        Update the dataset with new data after AL iteration
        Saves the updated dataset if save_data=True
        Args:
            X: array(N, 2) in the domain [0; 1]; states (i.e., grid positions)
            y: array(N, 1); scores at each position
        Return:
            DataLoader
        """

        states = X.clone()
        energies = y.clone()

        # append to in-memory dataset
        # TODO: also save a fraction to test_data?
        self.train_data.append(states, energies)

        if save_path is not None:
            readable_states = [self.env.state2readable(state) for state in states]
            readable_dataset = {
                "samples": readable_states,
                "energies": energies.tolist(),
            }
            import pandas as pd

            df = pd.DataFrame(readable_dataset)
            df.to_csv(save_path)

        return energies

    def get_custom_dataset(self, samples: torch.Tensor) -> Data:
        return GridData(torch.Tensor(samples), state2result=self.env.states2proxy)


class BraninDatasetHandler(GridDatasetHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_candidate_set(self, as_dataloader=False):
        import numpy as np

        # define candidate set
        xi = np.arange(0, self.env.length)
        yi = np.arange(0, self.env.length)
        grid = np.array(np.meshgrid(xi, yi))
        grid_flat = torch.tensor(grid.T, dtype=self.float).reshape(-1, 2)
        candidate_set = GridData(grid_flat, state2result=self.env.states2proxy)
        if as_dataloader:
            candidate_set = DataLoader(
                candidate_set,
                batch_size=self.batch_size,
            )
        return (
            candidate_set,
            xi,
            yi,
        )


class CandidateGridData(Dataset):
    # generates grid states based on the grid size, dimension of the grid, and 1-d index of the desired state
    # the grid has grid_size**dim states

    def __init__(self, grid_size, dim, step=1):
        self.grid_size = grid_size
        self.dim = dim
        self.step = step
        self.shape = (self.__len__(), self.dim)

    def __len__(self):
        return int(self.grid_size / self.step) ** self.dim

    def get_state_from_index(self, index):
        state = []
        for dim_i in range(self.dim):
            div = int(self.grid_size / self.step) ** (self.dim - dim_i - 1)
            state.append(
                (int(index / div) % int(self.grid_size / self.step)) * self.step
            )
        return torch.tensor(state)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.get_state_from_index(key)

        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            states = [self.get_state_from_index(i) for i in range(start, stop, step)]
            return torch.stack(states)

        states = [self.get_state_from_index(i) for i in key]
        return torch.stack(states)


class HartmannDatasetHandler(GridDatasetHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_candidate_set(self, step=1, as_dataloader=True):
        import numpy as np

        # define candidate set
        xi = np.arange(0, self.env.length)
        yi = np.arange(0, self.env.length)
        # grid = np.array(np.meshgrid(*[xi, yi] * 3))
        # grid_flat = torch.tensor(grid.T, dtype=torch.float64).reshape(-1, 6)
        grid_flat = CandidateGridData(self.env.length, self.env.n_dim, step=step)
        candidate_set = GridData(grid_flat, state2result=self.env.states2proxy)
        if as_dataloader:
            candidate_set = DataLoader(
                candidate_set,
                batch_size=self.batch_size,
            )
        return candidate_set, xi, yi
