from abc import ABC, abstractmethod
from datetime import datetime


class Logger(ABC):
    """
    Utils functions to compute and handle the statistics (saving them or send to
    wandb).
    """

    def __init__(self, project_name, run_name=None):
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
    def end(self):
        pass


class ConsoleLogger(Logger):
    def log_figure(self, figure, key):
        figure.show()

    def log_metric(self, value, key):
        print(key + ": " + value)

    def end(self):
        return


# class LocalLogger(Logger):
#     def __init__(self, project_name, run_name=None, log_dir=""):
#         super().__init__(project_name, run_name)
#         self.log_dir = log_dir

#     def log_figure(self, figure, key):


class WandBLogger(Logger):
    # TODO: this screws with other wandb runs...
    def __init__(self, project_name, run_name=None):
        super().__init__(project_name, run_name)
        import wandb

        self.wandb = wandb
        self.run = self.wandb.init(project=project_name, name=run_name)

    def log_figure(self, figure, key, step=None):
        figimg = self.wandb.Image(figure)
        self.run.log({key: figimg}, step=step)

    def log_metric(self, value, key, step=None):
        self.run.log({key: value}, step=step)

    def end(self):
        self.run.finish()
