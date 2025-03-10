from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


class PlotHelper:
    def __init__(self, device="cpu"):
        self.device = device

    def plot_function(
        self,
        fn,
        space,
        xi=None,
        yi=None,
        fig=None,
        ax=None,
        output_index=None,
        size_x=None,
        size_y=None,
        **kwargs,
    ):
        if size_x is None and size_y is None:
            size_x = int(len(space) ** (1 / 2))
            size_y = size_x
        elif size_x is None:
            size_x = len(space) / size_y
        elif size_y is None:
            size_y = len(space) / size_x
        # assert size_x * size_y == len(space)

        if xi is None:
            xi = np.arange(0, size_x)
        if yi is None:
            yi = np.arange(0, size_y)
        assert len(xi) == size_x and len(yi) == size_y

        if fig is None or ax is None:
            fig, ax = plt.subplots(nrows=1)

        # fn: function to plot
        # output_index: None if the output of the function is a single value; if the outputs are tuples index of the output that should be plotted

        if isinstance(space, torch.utils.data.dataloader.DataLoader):
            res = torch.Tensor([])
            for batch in space:
                batch_res = fn(batch.to(self.device)).to("cpu").detach()
                res = torch.concat([res, batch_res], dim=-1)

        else:
            res = fn(space.to(self.device)).to("cpu").detach()

        if output_index is not None:
            res = res[output_index]
        # ax.matshow(res)
        # https://matplotlib.org/stable/gallery/images_contours_and_fields/irregulardatagrid.html#sphx-glr-gallery-images-contours-and-fields-irregulardatagrid-py
        cntr = ax.contourf(
            xi,
            yi,
            res.reshape(size_x, size_y),
            levels=50,
        )
        fig.colorbar(cntr, ax=ax)
        return fig, ax

    def plot_samples(self, samples, ax=None, fig=None, targets=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(nrows=1)
        c = "red"
        if targets is not None:
            c = targets
        scatter = ax.scatter(
            x=samples[:, 1], y=samples[:, 0], c=c, marker="x", cmap="Reds"
        )
        if targets is not None:
            fig.colorbar(scatter, ax=ax)
        return fig, ax

    def plot_scores(self, **kwargs):
        pass

    def end(self, **kwargs):
        pass


class ProjectionPlotHelper(PlotHelper):
    def __init__(self, space, verbose=False):
        self.space = space
        super().__init__()

        from openTSNE import TSNE

        self.proj_fn = TSNE(2, verbose=verbose, random_state=31415)
        self.embedding = self.proj_fn.fit(space)

    def plot_function(self, fn, fig=None, ax=None, output_index=-1, **kwargs):
        if fig is None or ax is None:
            fig, ax = plt.subplots(nrows=1)

        # fn: function to plot
        # output_index: -1 if the output of the function is a single value; if the outputs are tuples index of the output that should be plotted
        res = fn(self.space)
        if output_index >= 0:
            res = res[output_index]
        res = res.to("cpu").detach()

        cntr = ax.tricontourf(
            self.embedding[:, 0],
            self.embedding[:, 1],
            res,
            levels=50,
        )  # , cmap="viridis_r")
        fig.colorbar(cntr, ax=ax)
        return fig, ax

    def plot_samples(self, samples, ax=None, fig=None, c=None, label=""):
        if fig is None or ax is None:
            fig, ax = plt.subplots(nrows=1)
        if c is None:
            c = "red"
        coords = self.embedding.transform(samples)
        ax.scatter(x=coords[:, 0], y=coords[:, 1], c=c, marker="x", label=label)
        return fig, ax


class CIME4RExportHelper(PlotHelper):
    def __init__(self, dataset_handler, device="cpu", logdir="logs/cime4r/"):
        super().__init__(device)

        self.dataset_handler = dataset_handler
        self.cime4r_df = pd.DataFrame()
        self.logdir = logdir
        self.init_dataframe()

    @abstractmethod
    def init_dataframe(self):
        pass

    def plot_function(
        self,
        fn,
        space,
        xi=None,
        yi=None,
        fig=None,
        ax=None,
        output_index=None,
        size_x=None,
        size_y=None,
        label=None,
        iteration=None,
    ):
        # fn: function to plot
        # output_index: None if the output of the function is a single value; if the outputs are tuples index of the output that should be plotted

        if isinstance(space, torch.utils.data.dataloader.DataLoader):
            assert len(space.dataset) == len(self.cime4r_df)
        else:
            assert len(space) == len(self.cime4r_df)
        res = fn(space)

        if output_index is not None:
            res = res[output_index]
        res = res.to("cpu").detach().tolist()

        if label is None:
            import time

            label = "col%i" % time.time()

        if iteration is not None:
            label = label + "_%i" % iteration

        self.cime4r_df[label] = res

        if fig is None or ax is None:
            return plt.subplots(ncols=1)
        return fig, ax

    def plot_samples(self, samples, ax=None, fig=None, targets=None):
        return fig, ax

    def plot_scores(self, selected_idcs, scores, i):
        self.cime4r_df.loc[selected_idcs.tolist(), "measured_target"] = scores.tolist()
        self.cime4r_df.loc[selected_idcs.tolist(), "experiment_cycle"] = i

    def end(self, filename):
        from pathlib import Path

        path = Path(self.logdir + "/cime4r/").resolve()
        path.mkdir(parents=True, exist_ok=True)
        self.cime4r_df.to_csv(path / filename, index=False)
        print("saved as", path / filename)


class BraninCIME4RExportHelper(CIME4RExportHelper):

    def init_dataframe(self):
        candidate_set, _, _ = self.dataset_handler.get_candidate_set()
        cols = ["state_%i_desc" % i for i in range(candidate_set.shape[1])]
        cime4r_df = pd.DataFrame(candidate_set[:], columns=cols)

        raw_state = candidate_set.get_raw_items()
        state_cols = []
        for i in range(raw_state.shape[1]):
            cime4r_df["state_%i" % i] = raw_state[:, i].to(torch.int)
            state_cols.append("state_%i" % i)

        cime4r_df["experiment_cycle"] = -1
        cime4r_df["measured_target"] = 0.0
        cime4r_df["SMILES_dummy"] = "*"
        cime4r_df = cime4r_df.set_index(state_cols)

        # init original train data
        train_x, train_y = self.dataset_handler.train_data.get_raw_items()
        train_idcs = [
            tuple([round(state[0], 1), round(state[1], 1)])
            for state in train_x.tolist()
        ]
        train_idcs, train_y.tolist()
        cime4r_df.loc[train_idcs, "measured_target"] = train_y.tolist()
        cime4r_df.loc[train_idcs, "experiment_cycle"] = 0

        self.cime4r_df = cime4r_df.reset_index()


class OCPCIME4RExportHelper(CIME4RExportHelper):

    def init_dataframe(self):
        shape = self.dataset_handler.candidate_data.shape

        candidate_set_dataloader, _, _ = self.dataset_handler.get_candidate_set(
            return_index=True
        )
        states = torch.Tensor([])
        idcs = torch.Tensor([])
        for batch, batch_idcs in candidate_set_dataloader:
            states = torch.concat([states, batch])
            idcs = torch.concat([idcs, batch_idcs])

        cols = ["state_%i_desc" % i for i in range(shape[1])]
        cime4r_df = pd.DataFrame(states, columns=cols)
        cime4r_df["experiment_cycle"] = -1
        cime4r_df["idx_in_dataset"] = idcs.squeeze().tolist()
        cime4r_df["measured_target"] = 1.0
        cime4r_df["SMILES_dummy"] = "*"
        cime4r_df.set_index("idx_in_dataset")

        # init original train data
        for i in range(len(self.dataset_handler.train_data)):
            datapoint = self.dataset_handler.train_data.get_raw_items(i)
            idx = datapoint.idx_in_dataset
            target = datapoint.y_relaxed
            cime4r_df.loc[idx, "measured_target"] = target
            cime4r_df.loc[idx, "experiment_cycle"] = 0

        self.cime4r_df = cime4r_df.reset_index()
