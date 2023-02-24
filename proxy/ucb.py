import torch
from .botorch_models import ProxyBotorchUCB
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling import SobolQMCNormalSampler
from .dropout_regressor import DropoutRegressor
import numpy as np
from regressor.regressor import DropoutRegressor as SurrogateDropoutRegressor
from utils.dkl_utils import batched_call


class UCB(DropoutRegressor):
    def __init__(self, kappa, **kwargs) -> None:
        super().__init__(**kwargs)
        self.kappa = kappa
        if not self.regressor.load_model():
            raise FileNotFoundError

    def __call__(self, inputs):
        # TODO: modify this. input arg would never be fids
        """
        Args
            inputs: batch x obs_dim
        Returns:
            score of dim (n_samples,), i.e, ndim=1"""
        if isinstance(inputs, np.ndarray):
            inputs = torch.FloatTensor(inputs).to(self.device)
        outputs = self.regressor.forward_with_uncertainty(
            inputs, self.num_dropout_samples
        )
        mean, std = torch.mean(outputs, dim=1), torch.std(outputs, dim=1)
        score = mean + self.kappa * std
        score = score.to(self.device).to(self.float)
        return score


class BotorchUCB(UCB):
    def __init__(self, sampler, **kwargs):
        super().__init__(**kwargs)
        self.sampler_config = sampler
        if isinstance(self.regressor, SurrogateDropoutRegressor):
            model = ProxyBotorchUCB(self.regressor, self.num_dropout_samples)
        else:
            model = self.regressor.surrogate
        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.sampler_config.num_samples]),
            seed=self.sampler_config.seed,
        )
        self.acqf = qUpperConfidenceBound(model=model, beta=self.kappa, sampler=sampler)
        self.out_dim = 1
        self.batch_size = 32

    def __call__(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, device=self.device, dtype=self.float)
        # inputs = inputs.unsqueeze(0)
        # acq_vals = torch.cat(
        #     batched_call(self.acq_fn, inputs, batch_size=1)
        # )
        # return acq_vals
        # TODO: call lanmguage_model.pool_features() --> line 292-304 in lambo.py
        inputs = inputs.unsqueeze(-2)
        acq_values = self.acqf(inputs)
        return acq_values
