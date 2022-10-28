import os
import torch
import torch.nn.functional as F
from abc import abstractmethod
from torch.nn.utils.rnn import pad_sequence
import numpy as np


def l2r(x):
    r_max = 0
    r_norm = 1
    score = torch.clip(x, min=-np.inf, max=r_max).sigmoid() / r_norm
    return score.cpu().detach()


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
        if self.config.acquisition.main.lower() == "proxy":
            self.acq = AcquisitionFunctionProxy(self.config, self.proxy)
        elif self.config.acquisition.main.lower() == "ucb":
            self.acq = AcquisitionFunctionUCB(self.config, self.proxy)
        elif self.config.acquisition.main.lower() == "ei":
            self.acq = AcquisitionFunctionEI(self.config, self.proxy)
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


class AcquisitionFunctionUCB(AcquisitionFunctionBase):
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
                outputs = (
                    torch.hstack(
                        [
                            self.proxy.model(inputs)
                            for _ in range(self.config.proxy.dropout_samples)
                        ]
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
            elif self.config.proxy.model.lower() == "lstm":
                outputs = (
                    torch.hstack(
                        [
                            self.proxy.model(inputs, input_afs_lens)
                            for _ in range(self.config.proxy.dropout_samples)
                        ]
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
            elif self.config.proxy.model.lower() == "transformer":
                outputs = (
                    torch.hstack(
                        [
                            self.proxy.model(inputs, None)
                            for _ in range(self.config.proxy.dropout_samples)
                        ]
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
        mean = np.mean(outputs, axis=1)
        std = np.std(outputs, axis=1)
        score = mean + self.config.acquisition.ucb.kappa * std
        score = l2r(torch.Tensor(score))
        score = score.unsqueeze(1)
        return score


class AcquisitionFunctionEI(AcquisitionFunctionBase):
    def __init__(self, config, proxy):
        super().__init__(config, proxy)

    def load_best_proxy(self):
        super().load_best_proxy()

    def getMinF(self, inputs, input_len, mask):
        # inputs = torch.Tensor(inputs).to(self.config.device)

        if self.config.proxy.model.lower() == "mlp":
            outputs = self.proxy.model(inputs)
        elif self.config.proxy.model.lower() == "lstm":
            outputs = self.proxy.model(inputs, input_len)
        elif self.config.proxy.model.lower() == "transformer":
            outputs = self.proxy.model(inputs, None)

        outputs = l2r(outputs)
        self.best_f = np.percentile(outputs, self.config.acquisition.ei.max_percentile)

    def get_reward_batch(self, inputs_af_base):  # inputs_af = list of ...
        super().get_reward_batch(inputs_af_base)
        inputs_af = list(map(torch.tensor, inputs_af_base))
        input_afs_lens = list(map(len, inputs_af_base))
        inputs = pad_sequence(inputs_af, batch_first=True, padding_value=0.0)

        self.getMinF(inputs, input_afs_lens, None)

        self.load_best_proxy()
        self.proxy.model.eval()
        with torch.no_grad():
            if self.config.proxy.model.lower() == "mlp":
                outputs = (
                    torch.hstack(
                        [
                            self.proxy.model(inputs)
                            for _ in range(self.config.proxy.dropout_samples)
                        ]
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
            elif self.config.proxy.model.lower() == "lstm":
                outputs = (
                    torch.hstack(
                        [
                            self.proxy.model(inputs, input_afs_lens)
                            for _ in range(self.config.proxy.dropout_samples)
                        ]
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
            elif self.config.proxy.model.lower() == "transformer":
                outputs = (
                    torch.hstack(
                        [
                            self.proxy.model(inputs, None)
                            for _ in range(self.config.proxy.dropout_samples)
                        ]
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
        outputs = l2r(
            torch.Tensor(outputs)
        )  # convert to tensor to use torch operations
        mean, std = torch.mean(outputs, dim=1), torch.std(outputs, dim=1)
        u = (mean - self.best_f) / (std + 1e-4)
        normal = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = std * (updf + u * ucdf)
        ei = ei.cpu().detach()
        return ei
