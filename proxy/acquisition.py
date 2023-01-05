from gflownet.proxy.base import Proxy
import numpy as np
import os
import torch
from .botorch_models import ProxyBotorchUCB
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling import SobolQMCNormalSampler


class DropoutRegressor(Proxy):
    def __init__(self, regressor, num_dropout_samples, model_path, device) -> None:
        super().__init__()
        self.regressor = regressor
        self.num_dropout_samples = num_dropout_samples
        self.model_path = model_path
        self.device = device

    def load_model(self):
        if os.path.exists(self.model_path):
            self.regressor.load_model(self.model_path)
        else:
            raise FileNotFoundError

    # TODO: Remove once PR38 is merged to gfn
    def state2proxy(self, state):
        # convert from oracle-friendly form to state as before PR38, state2oralce is performed on the state and then sent here
        state = state + 1
        state = state.astype(int)
        # state2proxy
        obs = np.zeros(6, dtype=np.float32)
        obs[(np.arange(len(state)) * 3 + state)] = 1
        return obs

    def preprocess_data(self, inputs):
        # TODO: Remove once PR38 is merged to gfn
        """
        Return proxy-friendly tensor on desired device"""
        inputs = torch.FloatTensor(list(map(self.state2proxy, inputs))).to(self.device)
        return inputs

    def __call__(self, inputs):
        """
        Args:
            inputs: proxy-compatible input tensor
            dim = n_samples x obs_dim

        Returns:
            vanilla rewards
                - (with no power/boltzmann) transformation
                - dim = n_samples
                - ndim = 1

        """
        self.load_model()
        inputs = self.preprocess_data(inputs)
        self.regressor.model.train()
        with torch.no_grad():
            output = self.regressor.model(inputs).detach().cpu().numpy().squeeze(-1)
        return output


class UCB(DropoutRegressor):
    def __init__(self, regressor, num_dropout_samples, model_path, kappa) -> None:
        super().__init__(regressor, model_path, num_dropout_samples)
        self.kappa = kappa

    def __call__(self, inputs):
        self.load_model()
        # TODO: Remove once PR38 is merged to gfn
        inputs = self.preprocess_data(inputs)
        self.regressor.model.train()
        outputs = self.regressor.forward_with_uncertainty(
            inputs, self.num_dropout_samples
        )
        mean, std = torch.mean(outputs, dim=1), torch.std(outputs, dim=1)
        score = mean + self.kappa * std
        score = torch.Tensor(score)
        score = score.unsqueeze(1)
        return score.detach().cpu().numpy()


class BotorchUCB(UCB):
    def __init__(self, regressor, num_dropout_samples, model_path, sampler, kappa):
        super().__init__(regressor, num_dropout_samples, model_path, kappa)
        self.sampler = SobolQMCNormalSampler(
            num_samples=sampler.num_samples,
            seed=sampler.seed,
            resample=sampler.resample,
        )

    def load_model(self):
        super().load_model()
        self.model = ProxyBotorchUCB(self.regressor, self.num_dropout_samples)

    def __call__(self, inputs):
        # TODO: Remove once PR38 is merged to gfn
        inputs = self.preprocess_data(inputs)
        UCB = qUpperConfidenceBound(
            model=self.model, beta=self.kappa, sampler=self.sampler
        )
        acq_values = UCB(inputs)
        return acq_values.detach().cpu().numpy()
