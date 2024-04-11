from abc import ABC, abstractmethod
from typing import Any
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
)


class AcquisitionHandler(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def fit(self, candidate_set, surrogate_model):
        # initialize the acquisition function self.acq_fn
        pass

    def __call__(self, candidate_set):
        return self.acq_fn(candidate_set)


botorch_acq_fn_mapper = {
    "qLowerBoundMaxValueEntropy": qLowerBoundMaxValueEntropy,
}


class BOTorchAcqHandler(AcquisitionHandler):
    def __init__(self, acq_fn_class="qLowerBoundMaxValueEntropy"):
        super().__init__()
        if type(acq_fn_class) == str:
            assert acq_fn_class in botorch_acq_fn_mapper, (
                "acq_fn_class should either be a function class or a string in "
                + botorch_acq_fn_mapper.keys()
            )
            acq_fn_class = botorch_acq_fn_mapper[acq_fn_class]
        self.acq_fn_class = acq_fn_class

    def fit(self, candidate_set, surrogate_model):
        self.acq_fn = self.acq_fn_class(
            surrogate_model,
            candidate_set=candidate_set,
        )
