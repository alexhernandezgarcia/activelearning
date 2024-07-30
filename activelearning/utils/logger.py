from gflownet.utils.logger import Logger
import torch
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb


class ActiveLearningLogger(Logger):
    """
    Utils functions to compute and handle the statistics (saving them or send to
    wandb). It can be passed on to surrogate, acquisition, sampler, ... to get the
    statistics of training of the generated data at real time
    """

    def __init__(
        self,
        config,
        surrogate_ckpts: str,
        **kwargs,
    ):
        super().__init__(config=config, **kwargs)
        # Checkpoints directory
        self.surrogate_ckpts_dir = self.logdir / surrogate_ckpts
        self.surrogate_ckpts_dir.mkdir(parents=True, exist_ok=True)

    def save_surrogate(self, model, optimizer, step: int = 1e9, final=False):
        if final:
            ckpt_id = "final"
            if self.debug:
                print(f"Saving final models in {self.surrogate_ckpts_dir}")
        else:
            ckpt_id = "_iter{:06d}".format(step)
            if self.debug:
                print(f"Saving models at step {step} in {self.surrogate_ckpts_dir}")

        if self.surrogate_ckpts_dir is not None:
            stem = Path(
                self.surrogate_ckpts_dir.stem
                + "_"
                + self.context
                + "_"
                + ckpt_id
                + ".ckpt"
            )
            path = self.surrogate_ckpts_dir.parent / stem
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                path,
            )
            if self.debug:
                print(f"Forward policy saved in {path}")

    def log_time_series(
        self,
        time_series: list,
        key,
        use_context=True,
        step=0,
        x_label="",
        y_label="",
        y_lim_min=None,
        y_lim_max=None,
    ):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(nrows=1)
        ax.plot(time_series)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_ylim(bottom=y_lim_min, top=y_lim_max)
        self.log_plots({key: fig}, step, use_context=use_context)
