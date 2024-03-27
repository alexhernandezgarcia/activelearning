import matplotlib.pyplot as plt
import numpy as np


class PlotHelper:
    def __init__(self, logger=None):
        self.logger = logger

    def plot_function(
        self,
        fn,
        space,
        xi=None,
        yi=None,
        fig=None,
        ax=None,
        output_index=-1,
        size_x=None,
        size_y=None,
    ):
        if size_x is None and size_y is None:
            size_x = int(len(space) ** (1 / 2))
            size_y = size_x
        elif size_x is None:
            size_x = len(space) / size_y
        elif size_y is None:
            size_y = len(space) / size_x
        assert size_x * size_y == len(space)

        if xi is None:
            xi = np.arange(0, size_x)
        if yi is None:
            yi = np.arange(0, size_y)
        assert len(xi) == size_x and len(yi) == size_y

        if fig is None or ax is None:
            fig, ax = plt.subplots(nrows=1)

        # fn: function to plot
        # output_index: -1 if the output of the function is a single value; if the outputs are tuples index of the output that should be plotted
        res = fn(space)
        if output_index >= 0:
            res = res[output_index]
        res = res.to("cpu").detach()
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

    def plot_samples(self, samples, ax=None, fig=None):
        if fig is None or ax is None:
            fig, ax = plt.subplots(nrows=1)

        ax.scatter(x=samples[:, 1], y=samples[:, 0], c="red", marker="x")
        return fig, ax

    def log_figure(self, fig, key="test"):
        if self.logger:
            figimg = self.logger.wandb.Image(fig)
            self.logger.wandb.log({key: figimg})
            plt.close(fig)
        else:
            fig.show()
