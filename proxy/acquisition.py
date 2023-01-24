import torch
from .botorch_models import ProxyBotorchUCB
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling import SobolQMCNormalSampler
from .dropout_regressor import DropoutRegressor


class UCB(DropoutRegressor):
    def __init__(self, regressor, num_dropout_samples, device, kappa) -> None:
        super().__init__(regressor, num_dropout_samples, device)
        self.kappa = kappa
        if not self.regressor.load_model():
            raise FileNotFoundError

    def __call__(self, inputs, fids):
        """
        Args
            inputs: batch x obs_dim
        Returns:
            score of dim (n_samples,), i.e, ndim=1"""
        inputs = torch.FloatTensor(inputs).to(self.device)
        outputs = self.regressor.forward_with_uncertainty(
            inputs, fids, self.num_dropout_samples
        )
        mean, std = torch.mean(outputs, dim=1), torch.std(outputs, dim=1)
        score = mean + self.kappa * std
        score = torch.Tensor(score).detach().cpu().numpy()
        return score


class BotorchUCB(UCB):
    def __init__(self, regressor, num_dropout_samples, device, kappa, sampler):
        super().__init__(regressor, num_dropout_samples, device, kappa)
        self.sampler_config = sampler
        if not self.regressor.load_model():
            raise FileNotFoundError
        self.model = ProxyBotorchUCB(self.regressor, self.num_dropout_samples)
        self.sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.sampler_config.num_samples]),
            seed=self.sampler_config.seed,
        )

    def __call__(self, inputs):
        inputs = torch.FloatTensor(inputs).to(self.device).unsqueeze(-2)
        UCB = qUpperConfidenceBound(
            model=self.model, beta=self.kappa, sampler=self.sampler
        )
        acq_values = UCB(inputs)
        return acq_values.detach().cpu().numpy()
