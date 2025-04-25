from typing import Optional

from gflownet.proxy.base import Proxy

from activelearning.acquisition.acquisition import Acquisition


class AcquisitionProxy(Proxy):
    def __init__(self, acquisition: Optional[Acquisition] = None, **kwargs):
        super().__init__(**kwargs)
        self.acquisition = acquisition

    def set_acquisition(self, acquisition: Acquisition):
        """Sets the Acquisition function of the proxy.

        Parameters
        ----------
        acquisition : Acquisition
            An Acquisition instance defining the acquisition function.
        """
        self.acquisition = acquisition

    def __call__(self, states):
        return self.acquisition(states)
