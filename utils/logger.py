from gflownet.utils.logger import Logger
import torch
from pathlib import Path
import numpy as np


class AL_Logger(Logger):
    """
    Utils functions to compute and handle the statistics (saving them or send to
    wandb). It can be passed on to querier, gfn, proxy, ... to get the
    statistics of training of the generated data at real time
    """

    def __init__(
        self,
        config,
        do,
        project_name,
        logdir,
        sampler,
        progress,
        lightweight,
        debug,
        proxy,
        run_name=None,
        tags=None,
    ):
        super().__init__(
            config,
            do,
            project_name,
            logdir,
            sampler,
            progress,
            lightweight,
            debug,
            run_name,
            tags,
        )
        self.proxy_period = (
            np.inf if proxy.period == None or proxy.period == -1 else proxy.period
        )

    def set_proxy_path(self, ckpt_id: str = None):
        if ckpt_id is None:
            self.proxy_ckpt_path = None
        else:
            self.proxy_ckpt_path = self.ckpts_dir / f"{ckpt_id}"

    def save_proxy(self, model, final, epoch):
        if not epoch % self.proxy_period or final:
            if final:
                ckpt_id = "final"
            else:
                ckpt_id = "epoch{:03d}".format(epoch)
            if self.proxy_ckpt_path is not None:
                stem = Path(
                    self.proxy_ckpt_path.stem + self.context + ckpt_id + ".ckpt"
                )
                path = self.proxy_ckpt_path.parent / stem
                torch.save(model.state_dict(), path)
