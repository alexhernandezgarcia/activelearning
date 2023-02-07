from botorch.acquisition.max_value_entropy_search import (
    qMultiFidelityLowerBoundMaxValueEntropy,
)
from gflownet.proxy.base import Proxy
from pathlib import Path
import pandas as pd
from .botorch_models import MultifidelityOracleModel, MultiFidelityProxyModel
import torch
import numpy as np
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility

from gpytorch.functions import inv_quad

CLAMP_LB = 1.0e-8
from botorch.models.utils import check_no_nans
from .botorch_models import FidelityCostModel


class MultiFidelityMES(Proxy):
    def __init__(
        self,
        num_dropout_samples,
        logger,
        env,
        is_model_oracle,
        n_fid,
        regressor=None,
        oracle=None,
        data_path=None,
        user_defined_cost=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.logger = logger
        self.is_model_oracle = is_model_oracle
        self.n_fid = n_fid
        self.oracle = oracle
        self.env = env
        self.user_defined_cost = user_defined_cost
        candidate_set = self.load_candidate_set()
        self.load_model(regressor, num_dropout_samples)
        self.get_cost_utility()
        if self.is_model_oracle:
            self.project = self.project_oracle_max_fidelity
        else:
            self.project = self.project_proxy_max_fidelity
        self.qMES = qMultiFidelityLowerBoundMaxValueEntropy(
            self.model,
            candidate_set=candidate_set,
            project=self.project,
            cost_aware_utility=self.cost_aware_utility,
        )

    def get_cost_utility(self):
        if self.user_defined_cost:
            # dict[fidelity] = cost
            self.cost_aware_utility = FidelityCostModel(
                fidelity_weights={0: 0.5, 1: 0.6, 2: 0.7},
                fixed_cost=0.1,
                device=self.device,
            )
        elif self.is_model_oracle:
            # target fidelity is 2.0
            cost_model = AffineFidelityCostModel(
                fidelity_weights={-1: 2.0}, fixed_cost=0.1
            )
            self.cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
        else:
            cost_model = AffineFidelityCostModel(
                fidelity_weights={-1: 1.0, -2: 0.0, -3: 0.0}, fixed_cost=0.1
            )
            self.cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
            # need to chnage subset of fidelity and target fidelity
            # subset is last three colums
            # target is [0 0 1] assuming fid=2 is the best fidelity

    def load_candidate_set(self):
        if self.is_model_oracle:
            path = self.data_path
            data = pd.read_csv(path, index_col=0)
            samples = data["samples"]
        else:
            path = self.logger.data_path.parent / Path("data_train.csv")
            data = pd.read_csv(path, index_col=0)
            samples = data["samples"]
            # state = [self.env.readable2state(sample) for sample in samples]
        state = [self.env.readable2state(sample) for sample in samples]
        state_proxy = self.env.statebatch2proxy(state)
        if isinstance(state_proxy, torch.Tensor):
            return state_proxy.to(self.device)
        return torch.FloatTensor(state_proxy).to(self.device)

    def load_model(self, regressor, num_dropout_samples):
        if self.is_model_oracle:
            self.model = MultifidelityOracleModel(
                self.oracle, self.n_fid, device=self.device
            )
        else:
            self.model = MultiFidelityProxyModel(
                regressor, num_dropout_samples, self.n_fid, device=self.device
            )

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

    def project_proxy_max_fidelity(self, states):
        input_dim = states.ndim
        max_fid = torch.zeros((states.shape[0], self.n_fid), device=self.device).long()
        max_fid[:, -1] = 1
        if input_dim == 3:
            states = states[:, :, : -self.n_fid]
            max_fid = max_fid.unsqueeze(1)
            states = torch.cat([states, max_fid], dim=2)
        elif input_dim == 2:
            states = states[:, : -self.n_fid]
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
