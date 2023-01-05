from gflownet.proxy.base import Proxy
import numpy as np
import os
import torch
from .botorch_models import ProxyBotorchUCB
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling import SobolQMCNormalSampler


class DropoutRegressor(Proxy):
    def __init__(self, regressor, num_dropout_samples, model_path) -> None:
        super().__init__()
        self.regressor = regressor
        self.num_dropout_samples = num_dropout_samples
        if os.path.exists(model_path):
            regressor.load_model(model_path)
        else:
            raise FileNotFoundError

    def __call__(self, inputs):
        """
        Args:
            inputs: proxy-compatible input tensor
        Returns:
            vanilla rewards (with no power/boltzmann) transformation

        """
        self.regressor.model.train()
        with torch.no_grad():
            mean, std, var = self.regressor.model(inputs)
        return mean, std, var


class UCB(DropoutRegressor):
    def __init__(self, regressor, num_dropout_samples, model_path, kappa) -> None:
        super().__init__(regressor, model_path, num_dropout_samples)
        self.kappa = kappa

    def __call__(self, inputs):
        self.regressor.model.train()
        outputs = self.regressor.forward_with_uncertainty(
            inputs, self.num_dropout_samples
        )
        mean, std = torch.mean(outputs, dim=1), torch.std(outputs, dim=1)
        score = mean + self.kappa * std
        score = torch.Tensor(score)
        score = score.unsqueeze(1)
        return score


class BotorchUCB(UCB):
    def __init__(self, regressor, num_dropout_samples, model_path, sampler, kappa):
        super().__init__(regressor, num_dropout_samples, model_path, kappa)
        self.model = ProxyBotorchUCB(regressor, self.num_dropout_samples)
        self.sampler = SobolQMCNormalSampler(
            num_samples=sampler.num_samples,
            seed=sampler.seed,
            resample=sampler.resample,
        )

    def __call__(self, inputs):
        UCB = qUpperConfidenceBound(
            model=self.model, beta=self.kappa, sampler=self.sampler
        )
        acq_values = UCB(inputs)
        return acq_values
