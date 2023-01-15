import torch
from .botorch_models import ProxyBotorchUCB
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling import SobolQMCNormalSampler
from .dropout_regressor import DropoutRegressor


class UCB(DropoutRegressor):
    def __init__(
        self, regressor, num_dropout_samples, model_path, device, kappa
    ) -> None:
        super().__init__(regressor, num_dropout_samples, model_path, device)
        self.kappa = kappa

    def __call__(self, inputs):
        """
        Args
            inputs: batch x obs_dim
        Returns:
            score of dim (n_samples,), i.e, ndim=1"""
        self.load_model()
        # TODO: Remove once PR38 is merged to gfn
        inputs = self.preprocess_data(inputs)
        outputs = self.regressor.forward_with_uncertainty(
            inputs, self.num_dropout_samples
        )
        mean, std = torch.mean(outputs, dim=1), torch.std(outputs, dim=1)
        score = mean + self.kappa * std
        score = torch.Tensor(score).detach().cpu().numpy()
        return score


class BotorchUCB(UCB):
    def __init__(
        self, regressor, num_dropout_samples, model_path, device, kappa, sampler
    ):
        super().__init__(regressor, num_dropout_samples, model_path, device, kappa)
        self.sampler_config = sampler

    def load_model(self):
        super().load_model()
        self.model = ProxyBotorchUCB(self.regressor, self.num_dropout_samples)
        self.sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.sampler_config.num_samples]),
            seed=self.sampler_config.seed,
        )

    def __call__(self, inputs):
        # TODO: Remove once PR38 is merged to gfn
        inputs = self.preprocess_data(inputs)
        self.load_model()
        UCB = qUpperConfidenceBound(
            model=self.model, beta=self.kappa, sampler=self.sampler
        )
        acq_values = UCB(inputs)
        return acq_values.detach().cpu().numpy()
