from typing import Callable, Optional, Union

import torch
from gflownet.envs.crystals.surface import CrystalSurface as CrystalSurfaceEnv
from ocpmodels.common.utils import make_trainer_from_dir
from ocpmodels.modules.normalizer import Normalizer
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from activelearning.dataset.dataset import Data, DatasetHandler


class OCPData(Data):
    """
    Implements the "Data" class for OCP data.

        data: DeupDataset - https://github.com/RolnickLab/ocp/blob/6a7ad7e41e6be6db9164586e4e944d72eeca9160/ocpmodels/datasets/lmdb_dataset.py#L254
            the target is given in the property "y_relaxed" and is a minimization task (smaller is better);
            when an item is returned, the target will be multiplied by -1 to turn it into a maximization task;
        target_normalizer: mean and std normalizer - https://github.com/RolnickLab/ocp/blob/c93899a23947cb7c1e1409cf6d7d7d8b31430bdd/ocpmodels/modules/normalizer.py#L11
        state2result: function that takes raw states (graph) (aka environment format) and transforms them into the desired format;
            in case of GFN environments, this can be the states2proxy function
        subset_idcs: specifies which subset of the data will be used (useful, if we want to have subsets for training and testing)
        return_target: specifies whether the target should be returned;
            if True, the target will be returned along the feature vector;
            the target will be turned into a maximization task by multiplying by -1;
        return_index: if True, the original index of an item will be returned along with the feature vector;
    """

    def __init__(
        self,
        data,
        target_normalizer,
        state2result=None,
        subset_idcs: torch.Tensor = None,
        float=torch.float64,
        return_target=True,
        return_index=False,
    ):
        if subset_idcs is None:
            subset_idcs = torch.arange(len(data))

        self.subset_idcs = subset_idcs
        self.float = float
        self.data = data
        self.state2result = state2result
        self.appended_data = []
        self.return_target = return_target
        self.return_index = return_index

        self.target_normalizer = target_normalizer

    @property
    def shape(self):
        return torch.Size([self.__len__(), self.data[0].deup_q.shape[-1]])

    def get_item_from_index(self, index):
        datapoint = self.get_raw_item(index)
        return (
            self.state2result(datapoint),
            torch.Tensor([datapoint.y_relaxed]),
            torch.Tensor([datapoint.idx_in_dataset]),
        )

    def __getitem__(self, key):
        if isinstance(key, int):
            x, y, idcs_in_dataset = self.get_item_from_index(key)
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            x = torch.Tensor([])
            y = torch.Tensor([])
            idcs_in_dataset = torch.Tensor([])
            for i in range(start, stop, step):
                x_i, y_i, idx_in_dataset = self.get_item_from_index(i)
                x = torch.concat([x, x_i])
                y = torch.concat([y, y_i])
                idcs_in_dataset = torch.concat([idcs_in_dataset, idx_in_dataset])
        else:
            x = torch.Tensor([])
            y = torch.Tensor([])
            idcs_in_dataset = torch.Tensor([])
            for i in key:
                x_i, y_i, idx_in_dataset = self.get_item_from_index(i)
                x = torch.concat([x, x_i])
                y = torch.concat([y, y_i])
                idcs_in_dataset = torch.concat([idcs_in_dataset, idx_in_dataset])

        x, y = self.preprocess(x.to(self.float).squeeze(), y.to(self.float).squeeze())
        if self.return_target:
            return x, y * -1  # turn into maximization problem
        else:
            if self.return_index:
                return x, idcs_in_dataset
            return x

    def get_raw_items(self, key: Union[int, slice, list] = None):
        if isinstance(key, int):
            return self.get_raw_item(key)

        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            states = [self.get_raw_item(i) for i in range(start, stop, step)]
            return states

        states = [self.get_raw_item(i) for i in key]
        return states

    def get_raw_item(self, index):
        if index < len(self.subset_idcs):
            # map to actual index of original dataset
            index = self.subset_idcs[index]
            return self.data[index]
        else:
            # if the index is higher than the subset_idcs length, the datapoint is located in the appended_data list
            index = index - len(self.subset_idcs)
            return self.appended_data[index]

    def __len__(self):
        return len(self.subset_idcs) + len(self.appended_data)

    def preprocess(self, X, y):
        return X, self.target_normalizer.norm(y).cpu()

    def append(self, X: Batch, y: torch.Tensor):
        y = self.target_normalizer.denorm(y).cpu()
        data_to_append = X  # .to_data_list()
        for i in range(len(data_to_append)):
            # overwriting the target value with the oracle value
            data_to_append[i].y_relaxed = y[i]

        self.appended_data.extend(data_to_append)


class OCPRawDataMapper(Dataset):
    def __init__(self, candidate_data: OCPData, subset_idcs: torch.Tensor):
        if subset_idcs is None:
            subset_idcs = torch.arange(len(candidate_data))
        self.subset_idcs = subset_idcs
        self.candidate_data = candidate_data

    def __len__(self):
        return len(self.subset_idcs)

    def __getitem__(self, key):
        return self.candidate_data.get_raw_items(self.subset_idcs[key])


class OCPDatasetHandler(DatasetHandler):

    def __init__(
        self,
        env: CrystalSurfaceEnv,  # TODO: which parts of this environment can we use in the dataset handler? are there any faenet processing functions etc?
        checkpoint_path,
        data_path,
        normalize_labels=True,
        target_mean=-1.525913953781128,
        target_std=2.279365062713623,
        train_fraction=1.0,
        batch_size=256,
        shuffle=True,
        float_precision: int = 64,
        # device="cpu",
    ):
        super().__init__(
            env=env,
            float_precision=float_precision,
            batch_size=batch_size,
            shuffle=shuffle,
        )

        self.train_fraction = train_fraction
        self.trainer = make_trainer_from_dir(
            checkpoint_path,
            mode="continue",
            overrides={
                "is_debug": True,
                "silent": True,
                "cp_data_to_tmpdir": False,
                "deup_dataset.create": False,
                "dataset": {
                    "default_val": "deup-val_ood_cat-val_ood_ads",
                    "deup-train-val_id": {
                        "src": data_path,
                        "normalize_labels": normalize_labels,
                        "target_mean": target_mean,
                        "target_std": target_std,
                    },
                    "deup-val_ood_cat-val_ood_ads": {"src": data_path},
                },
                "cpu": True,  # device == "cpu",
            },
            skip_imports=["qm7x", "gemnet", "spherenet", "painn", "comenet"],
            silent=True,
        )

        ocp_train_data = self.trainer.datasets["deup-train-val_id"]
        # if we specified a train_fraction, use a random subsample from the train data
        # as test data and don't use the test set at all
        if self.train_fraction < 1.0:
            index = torch.randperm(len(ocp_train_data))
            train_idcs = index[: int(len(ocp_train_data) * self.train_fraction)]
            test_idcs = index[int(len(ocp_train_data) * self.train_fraction) :]
            self.train_data = OCPData(
                ocp_train_data,
                state2result=self.state2proxy,
                subset_idcs=train_idcs,
                target_normalizer=self.trainer.normalizers["target"],
            )
            self.test_data = OCPData(
                ocp_train_data,
                state2result=self.state2proxy,
                subset_idcs=test_idcs,
                target_normalizer=self.trainer.normalizers["target"],
            )
            self.candidate_data = OCPData(
                ocp_train_data,
                state2result=self.state2proxy,
                # using all data instances as candidates in this case (uncomment, if we only want to use test set)
                # subset_idcs=test_idcs,
                return_target=False,
                target_normalizer=self.trainer.normalizers["target"],
            )
        else:
            self.train_data = OCPData(
                self.trainer.datasets["deup-train-val_id"],
                state2result=self.state2proxy,
                target_normalizer=self.trainer.normalizers["target"],
            )
            self.test_data = OCPData(
                self.trainer.datasets["deup-val_ood_cat-val_ood_ads"],
                state2result=self.state2proxy,
                target_normalizer=self.trainer.normalizers["target"],
            )
            self.candidate_data = OCPData(
                self.trainer.datasets["deup-val_ood_cat-val_ood_ads"],
                state2result=self.state2proxy,
                return_target=False,
                target_normalizer=self.trainer.normalizers["target"],
            )

    def state2proxy(self, state):
        hidden_states = state.deup_q
        return hidden_states.mean(0).unsqueeze(0)
        # since we only use one datapoint, we can just sum over this one, without the use of scatter
        # return scatter(hidden_states, states.batch, dim=0, reduce="mean")

    def maxY(self):
        # return 10  # -> TODO: what are the actual bounds?
        return 5  # when target is standard normalized
        # ... this takes too long
        # train_loader, _ = self.get_dataloader()
        # max_y = -torch.inf
        # for _, y in train_loader:
        #     max_i = y.max()
        #     if max_i > max_y:
        #         max_y = max_i
        # return max_y

    def minY(self):
        # return -10  # -> TODO: what are the actual bounds?
        return 5  # when target is standard normalized
        # ... this takes too long
        # train_loader, _ = self.get_dataloader()
        # min_y = torch.inf
        # for _, y in train_loader:
        #     min_i = y.min()
        #     if min_i < min_y:
        #         min_y = min_i
        # return min_y

    def get_dataloader(self):
        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=2,
            pin_memory=True,
        )
        test_loader = None
        if self.test_data is not None:
            test_loader = DataLoader(
                self.test_data,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=2,
                pin_memory=True,
            )

        return train_loader, test_loader

    def update_dataset(self, X: Batch, y: torch.Tensor, save_path=None):

        # append to in-memory dataset
        self.train_data.append(X, y.clone())

        if save_path is not None:
            # see https://github.com/RolnickLab/ocp/blob/main/ocpmodels/datasets/deup_dataset_creator.py#L371
            # for saving lmdb datasets
            print("TODO: save dataset to file system")

        return (
            y * -1
        )  # turn into maximization problem because the oracle returns minimization...

    def get_candidate_set(self, return_index=False, as_dataloader=True):
        if not as_dataloader:
            return self.candidate_data, None, None

        self.candidate_data.return_index = return_index
        test_loader = DataLoader(
            self.candidate_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        return test_loader, None, None

    def get_custom_dataset(self, samples):
        return OCPData(
            samples,
            state2result=self.state2proxy,
            target_normalizer=self.trainer.normalizers["target"],
            return_target=False,
        )

    """
    Transforms states into oracle format. 
    For ocp and faenet, we need the states with the faenet trainer's collate function.
    This results in a torch_geometric DataBatch object.
    """

    def states2oracle(self, samples) -> Batch:
        oracle_loader = DataLoader(
            samples,
            collate_fn=self.trainer.parallel_collater,
            num_workers=1,  # trainer.config["optim"]["num_workers"], # there ocurs a "AssertionError: can only test a child process" error when using several workers with cuda
            pin_memory=True,
            # batch_sampler=trainer.samplers["deup-train-val_id"],
            # batch_size=self.batch_size
            batch_size=len(samples),
        )
        return next(iter(oracle_loader))[0]

    def prepare_oracle_dataloader(self, dataset: OCPData, sample_idcs=None):
        candidate_set = OCPRawDataMapper(dataset, sample_idcs)
        loader = DataLoader(
            candidate_set,
            collate_fn=self.trainer.parallel_collater,
            num_workers=1,  # trainer.config["optim"]["num_workers"], # there ocurs a "AssertionError: can only test a child process" error when using several workers with cuda
            pin_memory=True,
            # batch_sampler=trainer.samplers["deup-train-val_id"],
            batch_size=self.batch_size,
        )
        return loader
