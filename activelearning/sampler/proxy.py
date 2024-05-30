from gflownet.proxy.base import Proxy


class AcquisitionProxy(Proxy):
    def __init__(self, acquisition, **kwargs):
        super().__init__(**kwargs)
        self.acquisition = acquisition

    def __call__(self, states):
        return self.acquisition(states)
