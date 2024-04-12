from abc import abstractmethod, ABC
from gflownet.utils.common import set_float_precision


class SurrogateModel(ABC):
    def __init__(self, float_precision=64, device="cpu", maximize=False, **kwargs):
        # use the kwargs for model specific configuration that is implemented in subclasses
        self.maximize = maximize
        self.target_factor = 1 if maximize else -1
        self.float = set_float_precision(float_precision)
        self.device = device

    @abstractmethod
    def fit(self, train_data):
        # fit the surrogate model
        pass

    @abstractmethod
    def get_predictions(self, states):
        pass
