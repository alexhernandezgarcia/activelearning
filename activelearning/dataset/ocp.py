from activelearning.dataset.dataset import DatasetHandler, Data
from torch.utils.data import DataLoader
import torch
from activelearning.utils.ocp import load_ocp_trainer


class OCPData(Data):
    def __init__(
        self,
        data,
        float=torch.float64,
        return_target=True,
    ):
        self.float = float
        self.data = data
        self.return_target = return_target

        self.shape = torch.Size([len(data), data[0].deup_q.shape[-1]])

    def get_item_from_index(self, index):
        datapoint = self.data[index]
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

    def __len__(self):
        return len(self.data)

    def preprocess(self, X, y):
        return X, y

    def append(self, X, y):
        # TODO: do we need this for ocp?
        pass


class OCPDatasetHandler(DatasetHandler):

    def __init__(
        self,
        checkpoint_path,
        batch_size=256,
        shuffle=True,
        float_precision: int = 64,
    ):
        super().__init__(
            float_precision=float_precision, batch_size=batch_size, shuffle=shuffle
        )

        self.trainer = load_ocp_trainer(checkpoint_path)

        self.train_data = OCPData(self.trainer.datasets["deup-train-val_id"])
        self.test_data = OCPData(self.trainer.datasets["deup-val_ood_cat-val_ood_ads"])
        self.candidate_data = OCPData(
            self.trainer.datasets["deup-val_ood_cat-val_ood_ads"], return_target=False
        )

    def maxY(self):
        return 10  # -> TODO: what are the actual bounds?
        # ... this takes too long
        # train_loader, _ = self.get_dataloader()
        # max_y = -torch.inf
        # for _, y in train_loader:
        #     max_i = y.max()
        #     if max_i > max_y:
        #         max_y = max_i
        # return max_y

    def minY(self):
        return -10  # -> TODO: what are the actual bounds?
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

    def update_dataset(self, X, y):
        print("for ocp there is no update strategy yet")

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
            samples.append(self.trainer.datasets["deup-val_ood_cat-val_ood_ads"][idx])

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
