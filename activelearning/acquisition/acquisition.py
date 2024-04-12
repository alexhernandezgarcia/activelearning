from abc import ABC, abstractmethod
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
)
from gflownet.utils.common import set_float_precision


class AcquisitionHandler(ABC):
    def __init__(self, device, float_precision, maximize=False):
        self.device = device
        self.float = set_float_precision(float_precision)
        self.target_factor = 1 if maximize else -1

    @abstractmethod
    def fit(self, candidate_set, surrogate_model):
        # initialize the acquisition function self.acq_fn
        pass

    def __call__(self, candidate_set):
        return (
            self.get_acquisition_values(candidate_set) * self.target_factor
        )  # TODO: remove self.target_factor; this will later be implemented directly in the gflownet environment (currently it always asumes negative values i.e. minimizing values)

    def get_acquisition_values(self, candidate_set):
        candidate_set = candidate_set.to(self.device).to(self.float)
        return self.acq_fn(candidate_set.unsqueeze(1)).detach()


botorch_acq_fn_mapper = {
    "qLowerBoundMaxValueEntropy": qLowerBoundMaxValueEntropy,
}


class BOTorchAcqHandler(AcquisitionHandler):
    def __init__(
        self,
        acq_fn_class="qLowerBoundMaxValueEntropy",
        device="cpu",
        float_precision=64,
        maximize=False,
    ):
        super().__init__(device, float_precision, maximize)
        if type(acq_fn_class) == str:
            assert acq_fn_class in botorch_acq_fn_mapper, (
                "acq_fn_class should either be a function class or a string in "
                + botorch_acq_fn_mapper.keys()
            )
            acq_fn_class = botorch_acq_fn_mapper[acq_fn_class]
        self.acq_fn_class = acq_fn_class

    def fit(self, candidate_set, surrogate_model):
        candidate_set = candidate_set.to(self.device).to(self.float)
        self.acq_fn = self.acq_fn_class(
            surrogate_model,
            candidate_set=candidate_set,
        )
