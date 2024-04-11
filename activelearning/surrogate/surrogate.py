from abc import abstractmethod, ABC
from gflownet.utils.common import set_float_precision


class SurrogatePipeline:
    def __init__(
        self, acq_fn, surrogate_model, float_precision=64, device="cpu", maximize=False
    ):
        # e.g.:
        #   acq_fn = botorch.acquisition.max_value_entropy_search.qLowerBoundMaxValueEntropy
        #   surrogate_model = surrogate.gp_surrogate.SingleTaskGPRegressor()
        self.maximize = maximize
        self.target_factor = 1 if maximize else -1
        self.float = set_float_precision(float_precision)
        self.device = device
        self.acq_fn = acq_fn
        self.surrogate_model = surrogate_model

    def fit(self, train_data):
        self.surrogate_model.fit(train_data)

    def get_predictions(self, states):
        states_proxy = states.clone().to(self.device).to(self.float)
        posterior = self.model.posterior(states_proxy)

        y_mean = posterior.mean
        y_std = posterior.variance.sqrt()
        y_mean = y_mean.squeeze(-1)
        y_std = y_std.squeeze(-1)
        return y_mean, y_std

    def setup(self, env):
        # TODO: what is this needed for? has to be present for gflownet
        pass

    def __call__(self, candidate_set):
        return (
            self.get_acquisition_values(candidate_set) * self.target_factor
        )  # TODO: remove self.target_factor; this will later be implemented directly in the gflownet environment (currently it always asumes negative values i.e. minimizing values)

    def get_acquisition_values(self, candidate_set):
        candidate_set = candidate_set.to(self.device).to(self.float)
        self.acq_fn.fit(
            surrogate_model=self.surrogate_model.model,
            candidate_set=candidate_set,
        )
        return self.acq_fn(candidate_set.unsqueeze(1)).detach()


class SurrogateModel(ABC):
    def __init__(self, float_precision=64, device="cpu", maximize=False, **kwargs):
        # use the kwargs for model specific configuration that is implemented in subclasses
        self.maximize = maximize
        self.target_factor = 1 if maximize else -1
        self.float = set_float_precision(float_precision)
        self.device = device

    @abstractmethod
    def posterior(self, states, train=False):
        # returns the posterior of the surrogate model for the given states
        pass

    @abstractmethod
    def fit(self, train_data):
        # fit the surrogate model
        pass
