from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from gflownet.utils.common import set_float_precision
from torch.nn.utils.rnn import pad_sequence
from abc import ABC, abstractmethod


class Data(Dataset):
    def __init__(self, X_data, y_data=None, device="cpu", float=torch.float64):
        self.float = float
        self.device = device
        self.X_data = X_data.to(self.float).to(self.device)
        self.y_data = None
        if y_data is not None:
            self.y_data = y_data.to(self.float).to(self.device)

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


class Branin_Data(Data):
    """
    Implements the "Data" class for Branin data. 
    
        X_data: array(N, 2) in the domain [0; grid_size]; states (i.e., grid positions) 
        y_data: array(N, 1); scores at each position
        normalize_scores: bool; maps the scores to [0; 1]
        grid_size: int; specifies the width and height of the Branin grid (grid_size x grid_size); used to normalize the state positions
        device: pytorch device used for calculations


    """
    def __init__(self, grid_size, X_data, y_data=None, normalize_scores=True, device="cpu", float=torch.float64):
        super().__init__(X_data, y_data, float=float, device=device)
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
        self.stats = self.get_statistics(self.y_data) # update the score statistics
    
    def preprocess(self, X, y):
        """
        - normalizes the scores
        - normalizes the states
        """
        states = X / self.grid_size
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

    



class DatasetHandler(ABC):
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


class BraninDatasetHandler(DatasetHandler):
    """
    loads initial dataset. a path must be passed from where the initial csv file is loaded. it also checks whether other iteration steps are present in the directory and loads them too.

    return dataset. the dataloader contains a preprocessing function that normalizes the data according to the Branin Grid environment.

    saves/appends future results. appends the oracle data to the dataset that is loaded in memory. additionally saves the results from the current iteration in a new csv file.
    """

    def __init__(
        self,
        grid_size,
        normalize_scores=True,
        train_fraction=0.8,
        batch_size=256,
        shuffle=True,
        train_path="storage/branin/sf/data_train.csv",
        test_path=None,
        device="cpu",
        float_precision=64,
    ):
        self.normalize_scores = normalize_scores
        self.train_fraction = train_fraction
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_path = train_path
        self.test_path = test_path
        self.device = device
        self.float = set_float_precision(float_precision)
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
        return str(state).replace("(", "[").replace(")", "]").replace(",", "")

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
            train_states = torch.stack([
                torch.tensor(self.readable2state(sample)) for sample in train_samples_X
            ])


        # load test data
        test_states = torch.Tensor()
        test_scores = torch.Tensor()
        if self.test_path is not None:
            test = pd.read_csv(self.test_path)
            test_samples_X = test["samples"].values.tolist()
            test_samples_y = test["energies"].values.tolist()
            test_scores = torch.tensor(test_samples_y)
            test_states = torch.stack([
                torch.tensor(self.readable2state(sample)) for sample in test_samples_X
            ])

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


        self.train_data = Branin_Data(
            self.grid_size,
            train_states, 
            train_scores,
            normalize_scores=self.normalize_scores,
            float=self.float,
            device=self.device
        )

        if len(test_states) > 0:
            self.test_data = Branin_Data(
                self.grid_size,
                test_states, 
                test_scores,
                normalize_scores=self.normalize_scores,
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
        # y = torch.tensor(y, dtype=self.float)  # , device=self.device
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


    def update_dataset(self, X, y):
        """
        Update the dataset with new data after AL iteration
        Saves the updated dataset if save_data=True
        Args:
            X: array(N, 2) in the domain [0; 1]; states (i.e., grid positions) 
            y: array(N, 1); scores at each position
        Return:
            DataLoader
        """

        states = X
        energies = y

        # append to in-memory dataset
        # TODO: also save a fraction to test_data?
        self.train_data.append(states, energies)

        # TODO: save the new datapoints in a new csv file
        readable_states = [self.state2readable(state) for state in states]
        readable_dataset = {
            "samples": readable_states,
            "energies": energies.tolist(),
        }

        return self.get_dataloader()


