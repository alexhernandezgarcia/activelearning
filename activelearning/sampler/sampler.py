from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch
from gflownet.utils.common import gflownet_from_config, set_device, set_float_precision

from activelearning.acquisition.acquisition import Acquisition


class Sampler(ABC):
    def __init__(
        self,
        acquisition: Acquisition,
        device: Union[str, torch.device],
        float_precision: Union[int, torch.dtype],
        **kwargs,
    ) -> None:
        self.acquisition = acquisition
        # Device
        self.device = set_device(device)
        # Float precision
        self.float_precision = set_float_precision(float_precision)

    @abstractmethod
    def get_samples(
        self,
        n_samples: int,
        candidate_set: Union[torch.utils.data.dataloader.DataLoader, torch.Tensor],
    ) -> Tuple[str, torch.Tensor]:
        pass

    def fit(self) -> None:
        pass


class GreedySampler(Sampler):
    """
    The Greedy Sampler class returns the top n samples according to the acquisition function.
    """

    def get_samples(self, n_samples, candidate_set):
        if isinstance(candidate_set, torch.utils.data.dataloader.DataLoader):
            acq_values = []
            for batch in candidate_set:
                acq_values.append(
                    self.acquisition(
                        batch.to(self.device).to(self.float_precision)
                    ).detach()
                )
            acq_values = torch.cat(acq_values)
            idx_pick = torch.argsort(acq_values, descending=True)[:n_samples]
            return (candidate_set.dataset.get_raw_items(idx_pick), idx_pick)
        else:
            acq_values = self.acquisition(
                candidate_set[:].to(self.device).to(self.float_precision)
            ).detach()
            idx_pick = torch.argsort(acq_values, descending=True)[:n_samples]
            return (candidate_set.get_raw_items(idx_pick), idx_pick)


class RandomSampler(Sampler):
    """
    The RandomSampler returns n random samples from a set of candidates.
    """

    def __init__(self, acquisition=None, device="cpu", float_precision=32, **kwargs):
        super().__init__(acquisition, device, float_precision)

    def get_samples(self, n_samples, candidate_set):
        if isinstance(candidate_set, torch.utils.data.dataloader.DataLoader):
            idx_pick = torch.randint(
                0, len(candidate_set.dataset), size=(n_samples,), device=self.device
            )
            return (candidate_set.dataset.get_raw_items(idx_pick), idx_pick)
        idx_pick = torch.randint(
            0, len(candidate_set), size=(n_samples,), device=self.device
        )
        return (candidate_set.get_raw_items(idx_pick), idx_pick)


class GFlowNetSampler(Sampler):
    """
    The GFlowNetSampler trains a GFlowNet in combination with a acquisition function.
    Then it generates n samples proportionally to the reward.
    """

    def __init__(self, env_maker, acquisition, conf, device, float_precision, **kwargs):
        super().__init__(acquisition, device, float_precision)

        # Set device and float precision in config
        conf.device = device
        conf.float_precision = float_precision

        # Initialize a GFlowNet sampler from the configuration file
        self.sampler = gflownet_from_config(conf)

    def fit(self):
        self.sampler.train()
        # self.sampler.logger.end()

    def get_samples(self, n_samples, candidate_set=None):
        batch, times = self.sampler.sample_batch(n_forward=n_samples, train=False)
        return (batch.get_terminating_states(), None)


class RandomGFlowNetSampler(Sampler):
    def __init__(self, env_maker, acquisition, conf, device, float_precision, **kwargs):
        super().__init__(acquisition, device, float_precision)
        import hydra

        self.env = env_maker()

    def get_samples(self, n_samples, candidate_set=None):
        if hasattr(self.env, "get_uniform_terminating_states"):
            samples = self.env.get_uniform_terminating_states(n_samples)
        else:
            samples = self.env.get_random_terminating_states(n_samples)

        return samples, None
