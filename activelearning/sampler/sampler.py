from abc import ABC, abstractmethod

import torch
from gflownet.utils.common import set_device, set_float_precision
from gflownet.utils.policy import parse_policy_config
from activelearning.acquisition.acquisition import Acquisition
from typing import Union, Tuple
import torch


class Sampler(ABC):
    def __init__(
        self,
        acquisition: Acquisition,
        device: Union[str, torch.device],
        float_precision: Union[int, torch.dtype],
        **kwargs
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

    def __init__(
        self,
        env_maker,
        acquisition,
        dataset_handler,
        conf,
        device,
        float_precision,
        **kwargs
    ):
        super().__init__(acquisition, device, float_precision)
        import hydra

        logger = hydra.utils.instantiate(
            conf.logger,
            conf,
            _recursive_=False,
        )

        env = env_maker()

        # The policy is used to model the probability of a forward/backward action
        forward_config = parse_policy_config(conf, kind="forward")
        backward_config = parse_policy_config(conf, kind="backward")

        forward_policy = hydra.utils.instantiate(
            forward_config,
            env=env,
            device=device,
            float_precision=float_precision,
        )
        backward_policy = hydra.utils.instantiate(
            backward_config,
            env=env,
            device=device,
            float_precision=float_precision,
        )

        # State flow
        if conf.state_flow is not None:
            state_flow = hydra.utils.instantiate(
                conf.state_flow,
                env=env,
                device=device,
                float_precision=float_precision,
                base=forward_policy,
            )
        else:
            state_flow = None

        reward = hydra.utils.instantiate(
            conf.proxy,
            device=device,
            float_precision=float_precision,
            acquisition=acquisition,
            dataset_handler=dataset_handler,
        )

        # GFlowNet Agent
        self.sampler = hydra.utils.instantiate(
            conf.agent,
            device=device,
            float_precision=float_precision,
            env_maker=env_maker,
            proxy=reward,
            forward_policy=forward_policy,
            backward_policy=backward_policy,
            state_flow=state_flow,
            logger=logger,
        )

    def fit(self):
        self.sampler.train()
        # self.sampler.logger.end()

    def get_samples(self, n_samples, candidate_set=None):
        batch, times = self.sampler.sample_batch(n_forward=n_samples, train=False)
        return (batch.get_terminating_states(), None)


class RandomGFlowNetSampler(Sampler):
    def __init__(self, env_maker, acquisition, conf, device, float_precision, **kwargs):
        super().__init__(acquisition, device, float_precision)

        self.env = env_maker()

    def get_samples(self, n_samples, candidate_set=None):
        if hasattr(self.env, "get_uniform_terminating_states"):
            samples = self.env.get_uniform_terminating_states(n_samples)
        else:
            samples = self.env.get_random_terminating_states(n_samples)

        return samples, None
