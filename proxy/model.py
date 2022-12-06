from gflownet.proxy.base import Proxy
import numpy as np
import os
# Surrogate would be the deep learning model that is trained to predict the scores
from surrogate import load_model
import torch

class Model(Proxy):
    def __init__(self, model_path) -> None:
        super().__init__()
        if os.path.exists(model_path):
            self.model = load_model(self.config.path.model_proxy)

        else:
            raise FileNotFoundError
    
    def __call__(self, inputs):
        self.model.train()
        with torch.no_grad():
            outputs = self.model(inputs)
        return outputs

# TODO: Move UCB to different file if need be
class UCB(Model):
    def __init__(self, model_path) -> None:
        super().__init__(model_path)
    
    def __call__(self, inputs):
        super().__call__(inputs)
        with torch.no_grad():
            outputs = (
                torch.hstack([self.proxy.model(inputs)])
                .cpu()
                .detach()
                .numpy()
            )

        mean = np.mean(outputs, axis=1)
        std = np.std(outputs, axis=1)
        score = mean + self.config.acquisition.ucb.kappa * std
        score = torch.Tensor(score)
        score = score.unsqueeze(1)
        return score