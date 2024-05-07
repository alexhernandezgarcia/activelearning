from activelearning.dataset.dataset import DatasetHandler, Data
from torch.utils.data import DataLoader
import torch
from torch_geometric.data import Batch
from ocpmodels.modules.normalizer import Normalizer
from ocpmodels.common.utils import make_trainer_from_dir


class OCPData(Data):
    def __init__(
        self,
        data,
        subset_idcs: torch.Tensor = None,
        float=torch.float64,
        return_target=True,
    ):
        """
        subset_idcs: specifies which subset of the data will be used (useful, if we want to have subsets for training and testing)
        """
        if subset_idcs is None:
            subset_idcs = torch.arange(len(data))

        self.subset_idcs = subset_idcs
        self.float = float
        self.data = data
        self.appended_data = []
        self.return_target = return_target

        # https://github.com/RolnickLab/ocp/blob/c93899a23947cb7c1e1409cf6d7d7d8b31430bdd/ocpmodels/trainers/base_trainer.py#L361
        # https://github.com/RolnickLab/ocp/blob/c93899a23947cb7c1e1409cf6d7d7d8b31430bdd/configs/models/tasks/is2re.yaml#L25
        # https://github.com/RolnickLab/ocp/blob/c93899a23947cb7c1e1409cf6d7d7d8b31430bdd/ocpmodels/modules/normalizer.py#L11
        self.target_normalizer = Normalizer(
            mean=-1.525913953781128,
            std=2.279365062713623,
            device="cpu",
        )

    @property
    def shape(self):
        return torch.Size([self.__len__(), self.data[0].deup_q.shape[-1]])

    def get_item_from_index(self, index):
        datapoint = self.get_raw_item(index)
        hidden_states = datapoint.deup_q
        # state_idcs = datapoint.batch
        # since we only use one datapoint, we can just sum over this one, without the use of scatter
        # return scatter(hidden_states, state_idcs, dim=0, reduce="add"), torch.Tensor(
        #     [datapoint.y_relaxed]
        # )
        return hidden_states.mean(0).unsqueeze(0), torch.Tensor([datapoint.y_relaxed])

    def __getitem__(self, key):
        if isinstance(key, int):
            x, y = self.get_item_from_index(key)
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            x = torch.Tensor([])
            y = torch.Tensor([])
            for i in range(start, stop, step):
                x_i, y_i = self.get_item_from_index(i)
                x = torch.concat([x, x_i])
                y = torch.concat([y, y_i])
        else:
            x = torch.Tensor([])
            y = torch.Tensor([])
            for i in key:
                x_i, y_i = self.get_item_from_index(i)
                x = torch.concat([x, x_i])
                y = torch.concat([y, y_i])

        x, y = self.preprocess(x.to(self.float).squeeze(), y.to(self.float).squeeze())
        if self.return_target:
            return x, y
        else:
            return x

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
        return X, self.target_normalizer.norm(y)

    def append(self, X: Batch, y: torch.Tensor):
        y = self.target_normalizer.denorm(y)
        data_to_append = X.to_data_list()
        # TODO: we should overwrite the target value with the oracle value, but the oracle value is off
        for i in range(len(data_to_append)):
            # overwriting the target value with the oracle value
            data_to_append[i].y_relaxed = y[i]

        self.appended_data.extend(data_to_append)


class OCPDatasetHandler(DatasetHandler):

    def __init__(
        self,
        checkpoint_path,
        data_path,
        train_fraction=1.0,
        batch_size=256,
        shuffle=True,
        float_precision: int = 64,
    ):
        super().__init__(
            float_precision=float_precision, batch_size=batch_size, shuffle=shuffle
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
                    "deup-train-val_id": {"src": data_path},
                    "deup-val_ood_cat-val_ood_ads": {"src": data_path},
                },
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
            self.train_data = OCPData(ocp_train_data, train_idcs)
            self.test_data = OCPData(ocp_train_data, test_idcs)
            self.candidate_data = OCPData(
                ocp_train_data,
                # using all data instances as candidates in this case (uncomment, if we only want to use test set)
                # test_idcs,
                return_target=False,
            )
        else:
            self.train_data = OCPData(self.trainer.datasets["deup-train-val_id"])
            self.test_data = OCPData(
                self.trainer.datasets["deup-val_ood_cat-val_ood_ads"]
            )
            self.candidate_data = OCPData(
                self.trainer.datasets["deup-val_ood_cat-val_ood_ads"],
                return_target=False,
            )

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
        self.train_data.append(X, y)

        if save_path is not None:
            # see https://github.com/RolnickLab/ocp/blob/main/ocpmodels/datasets/deup_dataset_creator.py#L371
            # for saving lmdb datasets
            print("TODO: save dataset to file system")

        return self.get_dataloader()

    def get_candidate_set(self):
        test_loader = DataLoader(
            self.candidate_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        return test_loader, None, None

    def prepare_dataset_for_oracle(self, samples, sample_idcs):
        if sample_idcs is None:
            return samples

        samples = []
        for idx in sample_idcs:
            samples.append(self.candidate_data.get_raw_item(idx))
            # samples.append(self.trainer.datasets["deup-val_ood_cat-val_ood_ads"][idx])

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
