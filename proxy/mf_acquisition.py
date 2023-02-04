from botorch.acquisition.max_value_entropy_search import (
    qMultiFidelityLowerBoundMaxValueEntropy,
)
from gflownet.proxy.base import Proxy
from pathlib import Path
import pandas as pd
from .botorch_models import MultifidelityOracleModel
import torch
import numpy as np
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility

from gpytorch.functions import inv_quad
CLAMP_LB = 1.0e-8
from botorch.models.utils import check_no_nans

class myGIBBON(qMultiFidelityLowerBoundMaxValueEntropy):

    def _compute_information_gain(
        self, X, mean_M, variance_M, covar_mM):
        posterior_m = self.model.posterior(
            X, observation_noise=True, posterior_transform=self.posterior_transform
        )
        mean_m = self.weight * posterior_m.mean.squeeze(-1)
        # batch_shape x 1
        variance_m = posterior_m.variance.clamp_min(CLAMP_LB).squeeze(-1)
        # batch_shape x 1
        check_no_nans(variance_m)

        # get stdv of noiseless variance
        stdv = variance_M.sqrt()
        # batch_shape x 1

        # define normal distribution to compute cdf and pdf
        normal = torch.distributions.Normal(
            torch.zeros(1, device=X.device, dtype=X.dtype),
            torch.ones(1, device=X.device, dtype=X.dtype),
        )

        # prepare max value quantities required by GIBBON
        mvs = torch.transpose(self.posterior_max_values, 0, 1)
        # 1 x s_M
        normalized_mvs = (mvs - mean_m) / stdv
        # batch_shape x s_M

        cdf_mvs = normal.cdf(normalized_mvs).clamp_min(CLAMP_LB)
        pdf_mvs = torch.exp(normal.log_prob(normalized_mvs))
        ratio = pdf_mvs / cdf_mvs
        check_no_nans(ratio)

        # prepare squared correlation between current and target fidelity
        rhos_squared = torch.pow(covar_mM.squeeze(-1), 2) / (variance_m * variance_M)
        # batch_shape x 1
        check_no_nans(rhos_squared)

        # calculate quality contribution to the GIBBON acqusition function
        inner_term = 1 - rhos_squared * ratio * (normalized_mvs + ratio)
        acq = -0.5 * inner_term.clamp_min(CLAMP_LB).log()
        # average over posterior max samples
        acq = acq.mean(dim=1).unsqueeze(0)

        if self.X_pending is None:
            # for q=1, no replusion term required
            return acq

        X_batches = torch.cat(
            [X, self.X_pending.unsqueeze(0).repeat(X.shape[0], 1, 1)], 1
        )
        # batch_shape x (1 + m) x d
        # NOTE: This is the blocker for supporting posterior transforms.
        # We would have to process this MVN, applying whatever operations
        # are typically applied for the corresponding posterior, then applying
        # the posterior transform onto the resulting object.
        V = self.model(X_batches)
        # Evaluate terms required for A
        A = V.lazy_covariance_matrix[:, 0, 1:].unsqueeze(1)
        # batch_shape x 1 x m
        # Evaluate terms required for B
        B = self.model.posterior(
            self.X_pending,
            observation_noise=True,
            posterior_transform=self.posterior_transform,
        ).mvn.covariance_matrix.unsqueeze(0)
        # 1 x m x m

        # use determinant of block matrix formula
        V_determinant = variance_m - inv_quad(B, A.transpose(1, 2)).unsqueeze(1)
        # batch_shape x 1

        # Take logs and convert covariances to correlations.
        r = V_determinant.log() - variance_m.log()
        r = 0.5 * r.transpose(0, 1)
        return acq + r


class MultiFidelityMESOracle(Proxy):
    def __init__(
        self, data_path, logger, env, is_model_oracle, n_fid, oracle=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.logger = logger
        self.is_model_oracle = is_model_oracle
        self.n_fid = n_fid
        self.oracle = oracle
        self.env = env
        candidate_set = self.load_candidate_set()
        self.load_model()
        self.get_cost_utility()
        if self.is_model_oracle:
            self.project = self.project_oracle_max_fidelity
        else:
            # TODO: implement for proxy, replace one-hot-encoding
            raise NotImplementedError("MF-GIBBON project() not adapted to proxy")
        self.qMES = myGIBBON(
            self.model,
            candidate_set=candidate_set,
            project=self.project,
            cost_aware_utility=self.cost_aware_utility,
        )

    def get_cost_utility(self):
        if self.is_model_oracle:
            # target fidelity is 2.0
            cost_model = AffineFidelityCostModel(
                fidelity_weights={2: 2.0}, fixed_cost=5.0
            )
            self.cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
        else:
            # need to chnage subset of fidelity and target fidelity
            # subset is last three colums
            # target is [0 0 1] assuming fid=2 is the best fidelity
            raise NotImplementedError
            # cost_model = AffineFidelityCostModel(fidelity_weights={len(): 1.0}, fixed_cost=5.0)

    def load_candidate_set(self):
        if self.is_model_oracle:
            path = self.data_path
            data = pd.read_csv(path, index_col=0)
        else:
            path = self.logger.logdir.root / Path("data") / self.data_path
            # TODO: if path is the train datsset of regressor
        samples = data["samples"]
        state = [self.env.readable2state(sample) for sample in samples]
        state_oracle = self.env.statebatch2proxy(state)
        # Tensor for grid
        # self.candidate_set = state_oracle
        return state_oracle

    def load_model(self):
        if self.is_model_oracle:
            self.model = MultifidelityOracleModel(
                self.oracle, self.n_fid, device=self.device
            )
        else:
            # TODO: if model is regressor
            raise NotImplementedError

    def project_oracle_max_fidelity(self, states):
        input_dim = states.ndim
        max_fid = torch.ones((states.shape[0], 1), device=self.device).long() * (
            self.n_fid - 1
        )
        if input_dim == 3:
            states = states[:, :, :-1]
            max_fid = max_fid.unsqueeze(1)
            states = torch.cat([states, max_fid], dim=2)
        elif input_dim == 2:
            states = states[:, :-1]
            states = torch.cat([states, max_fid], dim=1)
        return states

    def __call__(self, states):
        if isinstance(states, np.ndarray):
            states = torch.FloatTensor(states)
        # add extra dimension for q
        states = states.unsqueeze(-2)
        # qMES = qMultiFidelityLowerBoundMaxValueEntropy(
        #     self.model,
        #     candidate_set=self.candidate_set,
        #     project=self.project,
        #     cost_aware_utility=self.cost_aware_utility,
        # )
        
        mes = self.qMES(states)
        return mes
