import torch
from gflownet.utils.common import set_device, set_float_precision


class Selector:
    # Base Selector class just returns the first n samples from the candidate set
    def __init__(self, device, float_precision, **kwargs):
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)

    def __call__(self, n_samples, candidate_set):
        return candidate_set[-n_samples:]


class ScoreSelector(Selector):
    # Selector that takes a score function, which is used to select the top performing
    # samples (e.g., acquisition function)
    def __init__(self, score_fn, device="cpu", float_precision=64, maximize=True):
        super().__init__(device, float_precision)
        self.score_fn = score_fn
        self.maximize = maximize

    def __call__(self, n_samples, candidate_set):
        candidate_set = candidate_set.clone()
        scores = self.score_fn(candidate_set)
        idx_pick = torch.argsort(scores, descending=self.maximize)[:n_samples]
        return candidate_set[idx_pick]
