from gflownet.proxy.base import Proxy
import torch


class DropoutRegressor(Proxy):
    def __init__(self, regressor, num_dropout_samples, **kwargs) -> None:
        super().__init__(**kwargs)
        self.regressor = regressor
        self.num_dropout_samples = num_dropout_samples
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
        self.regressor.model.train()
        with torch.no_grad():
            # detach().cpu().numpy()
            output = self.regressor.model(inputs).squeeze(-1)
        return output
