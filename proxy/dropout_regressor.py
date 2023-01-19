from gflownet.proxy.base import Proxy
import torch


class DropoutRegressor(Proxy):
    def __init__(self, regressor, num_dropout_samples, device) -> None:
        super().__init__()
        self.regressor = regressor
        self.num_dropout_samples = num_dropout_samples
        self.device = device
        if not self.regressor.load_model():
            raise FileNotFoundError

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
        # self.load_model()
        inputs = torch.FloatTensor(inputs).to(self.device)
        self.regressor.model.train()
        with torch.no_grad():
            output = self.regressor.model(inputs).detach().cpu().numpy().squeeze(-1)
        return output
