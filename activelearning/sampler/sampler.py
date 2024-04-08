from abc import ABC, abstractmethod

import torch
from gflownet.utils.common import set_device, set_float_precision


class Sampler(ABC):
    def __init__(self, surrogate, device, float_precision, **kwargs):
        self.surrogate = surrogate
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)

    @abstractmethod
    def get_samples(self, n_samples, candidate_set):
        pass

    def fit(self):
        pass


class GreedySampler(Sampler):
    """
    The Greedy Sampler class returns the top n samples according to the acquisition function.
    """

    def get_samples(self, n_samples, candidate_set):
        candidate_set = candidate_set.clone().to(self.device)
        acq_values = self.surrogate.get_acquisition_values(candidate_set).detach()
        idx_pick = torch.argsort(acq_values)[-n_samples:]
        return candidate_set[idx_pick]


class RandomSampler(Sampler):
    """
    The RandomSampler returns n random samples from a set of candidates.
    """

    def __init__(self, surrogate=None):
        super().__init__(surrogate)

    def get_samples(self, n_samples, candidate_set):
        idx_pick = torch.randint(0, len(candidate_set), size=(n_samples,))
        return candidate_set[idx_pick]


class GFlowNetSampler(Sampler):
    """
    The GFlowNetSampler trains a GFlowNet in combination with a surrogate model.
    Then it generates n samples proportionally to the reward.
    """

    def __init__(self, surrogate, conf, device, float_precision):
        super().__init__(surrogate)
        import hydra

        logger = hydra.utils.instantiate(
            conf.logger,
            conf,
            _recursive_=False,
        )

        grid_env = hydra.utils.instantiate(
            conf.env,
            proxy=surrogate,
            device=device,
            float_precision=float_precision,
        )

        # The policy is used to model the probability of a forward/backward action
        forward_policy = hydra.utils.instantiate(
            conf.policy.forward,
            env=grid_env,
            device=device,
            float_precision=float_precision,
        )
        backward_policy = hydra.utils.instantiate(
            conf.policy.backward,
            env=grid_env,
            device=device,
            float_precision=float_precision,
        )

        # State flow
        if conf.state_flow is not None:
            state_flow = hydra.utils.instantiate(
                conf.state_flow,
                env=grid_env,
                device=device,
                float_precision=float_precision,
                base=forward_policy,
            )
        else:
            state_flow = None

        # GFlowNet Agent
        self.sampler = hydra.utils.instantiate(
            conf.agent,
            device=device,
            float_precision=float_precision,
            env=grid_env,
            forward_policy=forward_policy,
            backward_policy=backward_policy,
            state_flow=state_flow,
            buffer=conf.env.buffer,
            logger=logger,
        )

    def fit(self):
        self.sampler.train()
        # self.sampler.logger.end()

    def get_samples(self, n_samples, candidate_set=None):
        batch, times = self.sampler.sample_batch(n_forward=n_samples, train=False)
        return batch.get_terminating_states(proxy=True)
