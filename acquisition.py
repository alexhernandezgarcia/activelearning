import os
import torch
import torch.nn.functional as F
from abc import abstractmethod
from torch.nn.utils.rnn import pad_sequence


"""
ACQUISITION WRAPPER
"""


class AcquisitionFunction:
    def __init__(self, config, proxy):
        self.config = config
        self.proxy = proxy.proxy

        self.init_acquisition()

    def init_acquisition(self):
        # so far, only proxy acquisition function has been implemented, only add new acquisition class inheriting from AcquisitionFunctionBase to innovate
        if self.config.acquisition.main == "proxy":
            self.acq = AcquisitionFunctionProxy(self.config, self.proxy)
        else:
            raise NotImplementedError

    def get_reward(self, inputs_af_base):
        outputs = self.acq.get_reward_batch(inputs_af_base)
        return outputs


"""
BASE CLASS FOR ACQUISITION
"""


class AcquisitionFunctionBase:
    """
    Cf Oracle class : generic AF class which calls the right AF sub_class
    """

    @abstractmethod
    def __init__(self, config, proxy):
        self.config = config
        self.proxy = proxy
        self.device = self.config.device

        # the specific class of the exact AF is instantiated here

    @abstractmethod
    def load_best_proxy(self):
        """
        In case, loads the latest version of the proxy (no need normally)
        """
        if os.path.exists(self.config.path.model_proxy):
            self.proxy.load_model(self.config.path.model_proxy)

        else:
            raise FileNotFoundError

    @abstractmethod
    def get_reward_batch(self, inputs_af_base):
        """
        calls the get_reward method of the appropriate Acquisition Class (MUtual Information, Expected Improvement, ...)
        """
        pass


"""
SUBCLASS SPECIFIC ACQUISITION
"""


class AcquisitionFunctionProxy(AcquisitionFunctionBase):
    def __init__(self, config, proxy):
        super().__init__(config, proxy)

    def load_best_proxy(self):
        super().load_best_proxy()

    def get_reward_batch(self, inputs_af_base):  # inputs_af = list of ...
        super().get_reward_batch(inputs_af_base)

        inputs_af = list(map(torch.tensor, inputs_af_base))
        input_afs_lens = list(map(len, inputs_af_base))
        inputs = pad_sequence(inputs_af, batch_first=True, padding_value=0.0)

        self.load_best_proxy()
        self.proxy.model.eval()
        with torch.no_grad():
            if self.config.proxy.model.lower() == "mlp":
                outputs = self.proxy.model(inputs)
            elif self.config.proxy.model.lower() == "lstm":
                outputs = self.proxy.model(inputs, input_afs_lens)
            elif self.config.proxy.model.lower() == "transformer":
                outputs = self.proxy.model(inputs, None)

        return outputs
