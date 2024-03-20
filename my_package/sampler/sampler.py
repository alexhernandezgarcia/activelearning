from abc import abstractmethod
import torch


class Sampler:
    '''
    The base Sampler class returns the top n samples according to the acquisition function.
    '''
    def __init__(self, surrogate):
        self.surrogate = surrogate

    def get_samples(self, n_samples, candidate_set):
        acq_values = self.surrogate.get_acquisition_values(candidate_set)
        idx_pick = torch.argsort(acq_values)[-n_samples:]
        return candidate_set[idx_pick]

    @abstractmethod
    def fit(self):
        pass



class RandomSampler(Sampler):
    '''
    The RandomSampler returns n random samples from a set of candidates.
    '''
    def __init__(self, surrogate=None):
        super().__init__(surrogate)

    def get_samples(self, n_samples, candidate_set):
        idx_pick = torch.randint(0,len(candidate_set), size=(n_samples,))
        return candidate_set[idx_pick]
        

class GFlowNetSampler(Sampler):
    '''
    The GFlowNetSampler trains a GFlowNet in combination with a surrogate model.
    Then it generates n samples proportionally to the reward.
    '''
    def get_samples(self, n_samples, candidate_set):
        # TODO: implement
        raise NotImplementedError