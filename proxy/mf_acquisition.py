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
        self.load_candidate_set()
        self.load_model()
        self.get_cost_utility()
        if self.is_model_oracle:
            self.project = self.project_oracle_max_fidelity
        else:
            # TODO: implement for proxy, replace one-hot-encoding
            raise NotImplementedError("MF-GIBBON project() not adapted to proxy")

    def get_cost_utility(self):
        self.cost_aware_utility = None
        # cost_model = AffineFidelityCostModel(fidelity_weights={6: 1.0}, fixed_cost=5.0)
        # self.cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

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
        self.candidate_set = state_oracle

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
        qMES = qMultiFidelityLowerBoundMaxValueEntropy(
            self.model, candidate_set=self.candidate_set, project=self.project
        )
        mes = qMES(states)
        return mes
