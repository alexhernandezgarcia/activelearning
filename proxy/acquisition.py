import torch
from .botorch_models import ProxyBotorchUCB
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.sampling import SobolQMCNormalSampler
from .dropout_regressor import DropoutRegressor


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
