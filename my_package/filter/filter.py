import torch


class Filter:
    # Base Filter class just returns the first n samples from the candidate set
    def __init__(self):
        pass

    def __call__(self, n_samples, candidate_set):
        return candidate_set[:n_samples]
    

class OracleFilter(Filter):

    def __init__(self, oracle):
        super().__init__()
        self.oracle = oracle

    def __call__(self, n_samples, candidate_set, maximize=False):
        scores = self.oracle(candidate_set.clone())
        idx_pick = torch.argsort(scores, descending=maximize)[:n_samples]
        return candidate_set[idx_pick]