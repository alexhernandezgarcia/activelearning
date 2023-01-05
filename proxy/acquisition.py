from gflownet.proxy.base import Proxy
import numpy as np
import os
import torch


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
        self.model.train()
        with torch.no_grad():
            outputs = self.regressor.forward_with_uncertainty(
                inputs, self.num_dropout_samples
            )
        return outputs


class UCB(DropoutRegressor):
    def __init__(self, regressor, num_dropout_samples, model_path, kappa) -> None:
        super().__init__(regressor, model_path, num_dropout_samples)
        self.kappa = kappa

    def __call__(self, inputs):
        outputs = super().__call__(inputs)
        mean = np.mean(outputs, axis=1)
        std = np.std(outputs, axis=1)
        score = mean + self.kappa * std
        score = torch.Tensor(score)
        score = score.unsqueeze(1)
        return score
