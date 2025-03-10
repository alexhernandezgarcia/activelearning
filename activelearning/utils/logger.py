from abc import ABC, abstractmethod
from datetime import datetime

import matplotlib.pyplot as plt


class Logger(ABC):
    """
    Utils functions to compute and handle the statistics (saving them or send to
    wandb).
    """

    def __init__(self, project_name, run_name=None, **kwargs):
        self.project_name = project_name
        if run_name is None:
            date_time = datetime.today().strftime("%d/%m-%H:%M:%S")
            run_name = "{}".format(date_time)
        self.run_name = run_name

    @abstractmethod
    def log_figure(self, figure, key):
        pass

    @abstractmethod
    def log_metric(self, value, key):
        pass

    @abstractmethod
    def log_time_series(self, time_series: list, key):
        pass

    @abstractmethod
    def log_step(self, step):
        pass

    @abstractmethod
    def end(self):
        pass


class ConsoleLogger(Logger):
    def log_figure(self, figure, key, **kwargs):
        figure.show()

    def log_metric(self, value, key):
        print("%s: " % (key) + str(value))

    def log_time_series(self, time_series: list, key):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(nrows=1)
        ax.plot(time_series)
        self.log_figure(fig, key)
        # self.log_metric(time_series, key)

    def log_step(self, step):
        print("current step:", step)

    def end(self):
        return


# class LocalLogger(Logger):
#     def __init__(self, project_name, run_name=None, log_dir=""):
#         super().__init__(project_name, run_name)
#         self.log_dir = log_dir

#     def log_figure(self, figure, key):


class WandBLogger(Logger):
    # TODO: this screws with other wandb runs...
    def __init__(self, project_name, run_name=None, conf=None):
        super().__init__(project_name, run_name)
        import wandb

        self.wandb = wandb

        from omegaconf import OmegaConf

        wandb_config = None
        if conf is not None:
            wandb_config = OmegaConf.to_container(
                conf, resolve=True, throw_on_missing=True
            )
        self.run = self.wandb.init(
            project=project_name, name=run_name, config=wandb_config
        )
        self.log_dict = {}

    def log_figure(self, figure, key, step=None):
        figimg = self.wandb.Image(figure)
        self.log_dict[key] = figimg
        # self.run.log({key: figimg})

    def log_metric(self, value, key):
        self.log_dict[key] = value
        # self.run.log({key: value}, step=step)

    def log_time_series(self, time_series: list, key):
        fig, ax = plt.subplots(nrows=1)
        ax.plot(time_series)
        ax.set_ylabel("value")
        ax.set_xlabel("timestep")
        self.log_figure(ax, key)
        # self.log_dict[key] = plt
        # data = [[x, i] for i, x in enumerate(time_series)]
        # table = self.wandb.Table(data=data, columns=[key, "time_step"])
        # self.log_dict[key] = self.wandb.plot.line(table, key, "time_step", title=key)

    def log_step(self, step):
        self.run.log(self.log_dict, step)
        self.log_dict = {}

    def end(self):
        self.run.finish()
