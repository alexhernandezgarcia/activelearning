from activelearning.dataset.dataset import DatasetHandler, Data
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import torch


class GridData(Data):
    """
    Implements the "Data" class for Grid data.

        X_data: array(N, d) in the domain [0; grid_size], where d is the dimensionality of the grid world (e.g. for Branin d=2; for Hartmann d=6); states (i.e., grid positions)
        y_data: array(N, 1); scores at each position
        normalize_scores: bool; maps the scores to [0; 1]
        grid_size: int; specifies the width and height of the grid (grid_size x grid_size); used to normalize the state positions

    """

    def __init__(
        self,
        grid_size,
        X_data,
        y_data=None,
        normalize_scores=True,
        float=torch.float64,
    ):
        super().__init__(X_data, y_data, float=float)
        self.normalize_scores = normalize_scores
        self.grid_size = grid_size
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
        # X, y = self.deprocess(X, y) # append data in raw form (i.e., not normalized)
        X *= self.grid_size
        super().append(X, y)
        self.stats = self.get_statistics(self.y_data)  # update the score statistics

    def preprocess(self, X, y):
        """
        - normalizes the scores
        - normalizes the states
        """
        states = X / self.grid_size * 2 - 1  # normalize to [-1; 1]
        scores = None
        if self.normalize_scores and y is not None:
            scores = (y - self.stats["min"]) / (self.stats["max"] - self.stats["min"])

        return states, scores

    # def deprocess(self, X, y):
    #     """
    #     - denormalizes the scores
    #     - denormalizes the states
    #     """
    #     states = X * self.grid_size
    #     if self.normalize_scores and self.y_data is not None:
    #         scores = y * (self.stats["max"] - self.stats["min"]) + self.stats["min"]

    #     return states, scores


class GridDatasetHandler(DatasetHandler):
    """
    loads initial dataset. a path must be passed from where the initial csv file is loaded. it also checks whether other iteration steps are present in the directory and loads them too.

    return dataset. the dataloader contains a preprocessing function that normalizes the data according to the Grid environment.

    saves/appends future results. appends the oracle data to the dataset that is loaded in memory. additionally saves the results from the current iteration in a new csv file.
    """

    def __init__(
        self,
        grid_size,
        normalize_scores=True,
        train_fraction=0.8,
        batch_size=256,
        shuffle=True,
        train_path=None,
        test_path=None,
        float_precision=64,
    ):
        super().__init__(
            float_precision=float_precision, batch_size=batch_size, shuffle=shuffle
        )

        self.normalize_scores = normalize_scores
        self.train_fraction = train_fraction
        self.train_path = train_path
        self.test_path = test_path
        self.grid_size = grid_size

        self.initialise_dataset()

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
        return [float(el) for el in readable.strip("[]").split(" ") if el != ""]

    def state2readable(self, state):
        """
        Converts a state (a list of positions) into a human-readable string
        representing a state.
        """
        return (
            str(state.to(torch.int16).tolist())
            .replace("(", "[")
            .replace(")", "]")
            .replace(",", "")
        )

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
                    torch.tensor(self.readable2state(sample))
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
                [torch.tensor(self.readable2state(sample)) for sample in test_samples_X]
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
            self.grid_size,
            train_states,
            train_scores,
            normalize_scores=self.normalize_scores,
            float=self.float,
        )

        if len(test_states) > 0:
            self.test_data = GridData(
                self.grid_size,
                test_states,
                test_scores,
                normalize_scores=self.normalize_scores,
                float=self.float,
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
            readable_states = [self.state2readable(state) for state in states]
            readable_dataset = {
                "samples": readable_states,
                "energies": energies.tolist(),
            }
            import pandas as pd

            df = pd.DataFrame(readable_dataset)
            df.to_csv(save_path)

        return self.get_dataloader()


class BraninDatasetHandler(GridDatasetHandler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_candidate_set(self, as_dataloader=False):
        import numpy as np

        # define candidate set
        xi = np.arange(0, self.grid_size)
        yi = np.arange(0, self.grid_size)
        grid = np.array(np.meshgrid(xi, yi))
        grid_flat = torch.tensor(grid.T, dtype=self.float).reshape(-1, 2)
        candidate_set = GridData(self.grid_size, grid_flat)[:]
        if as_dataloader:
            candidate_set = DataLoader(
                candidate_set,
                batch_size=self.batch_size,
            )
        return (
            candidate_set,
            xi / self.grid_size * 2 - 1,
            yi / self.grid_size * 2 - 1,
        )  # scale to [-1; 2]


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
        xi = np.arange(0, self.grid_size, step)
        yi = np.arange(0, self.grid_size, step)
        # grid = np.array(np.meshgrid(*[xi, yi] * 3))
        # grid_flat = torch.tensor(grid.T, dtype=torch.float64).reshape(-1, 6)
        grid_flat = CandidateGridData(self.grid_size, 6, step=step)
        candidate_set = GridData(self.grid_size, grid_flat)
        if as_dataloader:
            candidate_set = DataLoader(
                candidate_set,
                batch_size=self.batch_size,
            )
        return candidate_set, xi / self.grid_size, yi / self.grid_size
