from gflownet.envs.base import GFlowNetEnv as BaseGFlowNetEnv
from abc import abstractmethod, ABC


class Envrionment(ABC):
    @abstractmethod
    def initialize_dataset():
        pass


class GFlowNetEnv(Envrionment, BaseGFlowNetEnv):
    def __init__(
        self, proxy_state_format, beta_factor=0, norm_factor=1, proxy=None, **kwargs
    ):
        super().__init__(proxy=proxy, **kwargs)
        self.proxy_state_format = proxy_state_format
        self.beta_factor = beta_factor
        self.norm_factor = norm_factor
        if proxy is not None:
            self.set_proxy(proxy)

    def set_proxy(self, proxy):
        self.proxy = proxy
        if hasattr(self, "proxy_factor"):
            return
        if self.proxy is not None and self.proxy.maximize is not None:
            # can be None for dropout regressor/UCB
            maximize = self.proxy.maximize
        elif self.oracle is not None:
            maximize = self.oracle.maximize
        else:
            raise ValueError("Proxy and Oracle cannot be None together.")
        if maximize:
            self.proxy_factor = 1.0
        else:
            self.proxy_factor = -1.0
    
