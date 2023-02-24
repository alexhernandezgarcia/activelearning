import torch
from .botorch_models import ProxyBotorchUCB
from botorch.acquisition.monte_carlo import (
    MCAcquisitionFunction,
    qExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.sampling import SobolQMCNormalSampler
from .dropout_regressor import DropoutRegressor
import numpy as np
from regressor.regressor import DropoutRegressor as SurrogateDropoutRegressor
from utils.dkl_utils import batched_call
from botorch.utils.transforms import (
    concatenate_pending_points,
    match_batch_shape,
    t_batch_mode_transform,
)
import math

# class qUpperConfidenceBound(MCAcquisitionFunction):
#     r"""MC-based batch Upper Confidence Bound.
#     Uses a reparameterization to extend UCB to qUCB for q > 1 (See Appendix A
#     of [Wilson2017reparam].)
#     `qUCB = E(max(mu + |Y_tilde - mu|))`, where `Y_tilde ~ N(mu, beta pi/2 Sigma)`
#     and `f(X)` has distribution `N(mu, Sigma)`.
#     Example:
#         >>> model = SingleTaskGP(train_X, train_Y)
#         >>> sampler = SobolQMCNormalSampler(1024)
#         >>> qUCB = qUpperConfidenceBound(model, 0.1, sampler)
#         >>> qucb = qUCB(test_X)
#     """

#     def __init__(
#         self,
#         model,
#         beta,
#         sampler=None,
#         objective=None,
#         posterior_transform=None,
#         X_pending=None,
#     ) -> None:
#         r"""q-Upper Confidence Bound.
#         Args:
#             model: A fitted model.
#             beta: Controls tradeoff between mean and standard deviation in UCB.
#             sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
#                 more details.
#             objective: The MCAcquisitionObjective under which the samples are
#                 evaluated. Defaults to `IdentityMCObjective()`.
#             posterior_transform: A PosteriorTransform (optional).
#             X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that have
#                 points that have been submitted for function evaluation but have not yet
#                 been evaluated. Concatenated into X upon forward call. Copied and set to
#                 have no gradient.
#         """
#         super().__init__(
#             model=model,
#             sampler=sampler,
#             objective=objective,
#             posterior_transform=posterior_transform,
#             X_pending=X_pending,
#         )
#         self.beta_prime = math.sqrt(beta * math.pi / 2)

#     @concatenate_pending_points
#     @t_batch_mode_transform()
#     def forward(self, X):
#         r"""Evaluate qUpperConfidenceBound on the candidate set `X`.
#         Args:
#             X: A `batch_sahpe x q x d`-dim Tensor of t-batches with `q` `d`-dim design
#                 points each.
#         Returns:
#             A `batch_shape'`-dim Tensor of Upper Confidence Bound values at the given
#             design points `X`, where `batch_shape'` is the broadcasted batch shape of
#             model and input `X`.
#         """
#         posterior = self.model.posterior(
#             X=X, posterior_transform=self.posterior_transform
#         )
#         samples = self.get_posterior_samples(posterior)
#         obj = self.objective(samples, X=X)
#         mean = obj.mean(dim=0)
#         ucb_samples = mean + self.beta_prime * (obj - mean).abs()
#         return ucb_samples.max(dim=-1)[0].mean(dim=0)


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
        # self.acqf = qExpectedImprovement(
        #     model=model,
        #     best_f=0.9,
        #     sampler=sampler,
        # )
        self.out_dim = 1
        self.batch_size = 32

    def __call__(self, inputs):
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, device=self.device, dtype=self.float)
        if isinstance(self.regressor, SurrogateDropoutRegressor) == False:
            if inputs[0][0] != self.regressor.tokenizer.bos_idx:
                input_tok = self.regressor.tokenizer.transform(inputs)
            else:
                input_tok = inputs
            (
                input_tok_features,
                input_mask,
            ) = self.regressor.language_model.get_token_features(input_tok)
            _, pooled_features = self.regressor.language_model.pool_features(
                input_tok_features, input_mask
            )
            inputs = pooled_features
        inputs = inputs.unsqueeze(-2)
        acq_values = self.acqf(inputs)
        return acq_values
