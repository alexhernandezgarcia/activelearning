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
from abc import abstractmethod

# class MultiFidelityMES(Proxy):
#     def __init__(
#         self,
#         num_dropout_samples,
#         logger,
#         env,
#         is_model_oracle,
#         n_fid,
#         fixed_cost,
#         regressor=None,
#         oracle=None,
#         data_path=None,
#         user_defined_cost=True,
#         **kwargs
#     ):
#         super().__init__(**kwargs)
#         # self.data_path = data_path
#         self.logger = logger
#         self.is_model_oracle = is_model_oracle
#         self.n_fid = n_fid #base
#         # self.oracle = oracle
#         self.env = env
#         self.user_defined_cost = user_defined_cost #base
#         self.fixed_cost = 1e-6 #base
#         # TODO: remove member variable candidate_set as
#         self.candidate_set = self.load_candidate_set() #base
#         self.load_model(regressor, num_dropout_samples) #base but check args
#         self.get_cost_utility()
#         if self.is_model_oracle:
#             self.project = self.project_oracle_max_fidelity
#         else:
#             self.project = self.project_proxy_max_fidelity
#         # base
#         self.qMES = qMultiFidelityLowerBoundMaxValueEntropy(
#             self.model,
#             candidate_set=self.candidate_set,
#             project=self.project,
#             cost_aware_utility=self.cost_aware_utility,
#         )

#     def get_cost_utility(self):
#         if self.user_defined_cost:
#             # dict[fidelity] = cost
#             self.cost_aware_utility = FidelityCostModel(
#                 fidelity_weights=self.env.fidelity_costs,
#                 fixed_cost=self.fixed_cost,
#                 device=self.device,
#             )
#         elif self.is_model_oracle:
#             # target fidelity is 2.0
#             cost_model = AffineFidelityCostModel(
#                 fidelity_weights={-1: 2.0}, fixed_cost=self.fixed_cost
#             )
#             self.cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
#         else:
#             cost_model = AffineFidelityCostModel(
#                 fidelity_weights={-1: 1.0, -2: 0.0, -3: 0.0}, fixed_cost=self.fixed_cost
#             )
#             self.cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

#     def load_candidate_set(self):
#         if self.is_model_oracle:
#             path = self.data_path
#             data = pd.read_csv(path, index_col=0)
#             samples = data["samples"]
#         else:
#             path = self.logger.data_path.parent / Path("data_train.csv")
#             data = pd.read_csv(path, index_col=0)
#             samples = data["samples"]
#         state = [self.env.readable2state(sample) for sample in samples]
#         state_proxy = self.env.statebatch2proxy(state)
#         if isinstance(state_proxy, torch.Tensor):
#             return state_proxy.to(self.device)
#         return torch.FloatTensor(state_proxy).to(self.device)

#     def load_model(self, regressor, num_dropout_samples):
#         if self.is_model_oracle:
#             self.model = MultifidelityOracleModel(
#                 self.oracle, self.n_fid, device=self.device
#             )
#         else:
#             self.model = MultiFidelityProxyModel(
#                 regressor, num_dropout_samples, self.n_fid, device=self.device
#             )

#     def project_oracle_max_fidelity(self, states):
#         input_dim = states.ndim
#         max_fid = torch.ones((states.shape[0], 1), device=self.device).long() * (
#             self.n_fid - 1
#         )
#         if input_dim == 3:
#             states = states[:, :, :-1]
#             max_fid = max_fid.unsqueeze(1)
#             states = torch.cat([states, max_fid], dim=2)
#         elif input_dim == 2:
#             states = states[:, :-1]
#             states = torch.cat([states, max_fid], dim=1)
#         return states

#     def project_proxy_max_fidelity(self, states):
#         input_dim = states.ndim
#         max_fid = torch.zeros((states.shape[0], self.n_fid), device=self.device).long()
#         max_fid[:, -1] = 1
#         if input_dim == 3:
#             states = states[:, :, : -self.n_fid]
#             max_fid = max_fid.unsqueeze(1)
#             states = torch.cat([states, max_fid], dim=2)
#         elif input_dim == 2:
#             states = states[:, : -self.n_fid]
#             states = torch.cat([states, max_fid], dim=1)
#         return states

#     def __call__(self, states):
#         if isinstance(states, np.ndarray):
#             states = torch.FloatTensor(states)
#         # add extra dimension for q
#         states = states.unsqueeze(-2)
#         # qMES = qMultiFidelityLowerBoundMaxValueEntropy(
#         #     self.model,
#         #     candidate_set=self.candidate_set,
#         #     project=self.project,
#         #     cost_aware_utility=self.cost_aware_utility,
#         # )

#         mes = self.qMES(states)
#         return mes


class MultiFidelityMES(Proxy):
    def __init__(self, env, cost_type, fixed_cost, logger, **kwargs):
        super().__init__(**kwargs)
        self.env = env
        self.cost_type = cost_type
        self.fixed_cost = fixed_cost
        self.n_fid = len(env.fidelity_costs)
        candidate_set = self.load_candidate_set()
        cost_aware_utility = self.get_cost_utility()
        model = self.load_model()
        self.logger = logger
        self.qMES = qMultiFidelityLowerBoundMaxValueEntropy(
            model,
            candidate_set=candidate_set,
            project=self.project,
            cost_aware_utility=cost_aware_utility,
        )

    @abstractmethod
    def project(self, states):
        pass

    @abstractmethod
    def load_candidate_set(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    def get_cost_utility(self):
        if self.cost_type == "user":
            cost_aware_utility = FidelityCostModel(
                fidelity_weights=self.env.fidelity_costs,
                fixed_cost=self.fixed_cost,
                device=self.device,
            )
        elif self.cost_type == "botorch":
            cost_model = AffineFidelityCostModel(
                fidelity_weights={-1: 2.0}, fixed_cost=self.fixed_cost
            )
            cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
        elif self.cost_type == "botorch_proxy":
            cost_model = AffineFidelityCostModel(
                fidelity_weights={-1: 1.0, -2: 0.0, -3: 0.0}, fixed_cost=self.fixed_cost
            )
            cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
        else:
            raise NotImplementedError(
                "Cost Type <{}> not implemented. Possible options are 'user', 'botorch', 'botorch_proxy'.".format(
                    self.cost_type
                )
            )
        return cost_aware_utility

    def __call__(self, states):
        if isinstance(states, np.ndarray):
            states = torch.FloatTensor(states)
        # add extra dimension for q
        states = states.unsqueeze(-2)
        mes = self.qMES(states)
        return mes

    def plot_context_points(self, **kwargs):
        pass


class ProxyMultiFidelityMES(MultiFidelityMES):
    def __init__(self, regressor, num_dropout_samples, **kwargs):
        self.regressor = regressor
        self.num_dropout_samples = num_dropout_samples
        super().__init__(**kwargs)

    def load_model(self):
        model = MultiFidelityProxyModel(
            self.regressor, self.num_dropout_samples, self.n_fid, device=self.device
        )
        return model

    def project(self, states):
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


class OracleMultiFidelityMES(MultiFidelityMES):
    def __init__(self, data_path, oracle, **kwargs):
        self.data_path = data_path
        self.oracle = oracle
        super().__init__(**kwargs)

    def load_candidate_set(self):
        path = self.data_path
        data = pd.read_csv(path, index_col=0)
        samples = data["samples"]
        state = [self.env.readable2state(sample) for sample in samples]
        state_proxy = self.env.statebatch2proxy(state)
        if isinstance(state_proxy, torch.Tensor):
            return state_proxy.to(self.device)
        return torch.FloatTensor(state_proxy).to(self.device)

    def load_model(self):
        model = MultifidelityOracleModel(self.oracle, self.n_fid, device=self.device)
        return model

    def project(self, states):
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

    def plot_context_points(self):
        path = self.data_path
        data = pd.read_csv(path, index_col=0)
        samples = data["samples"]
        state = [self.env.readable2state(sample) for sample in samples]
        state_proxy = self.env.statebatch2proxy(state)
        scores = self(state_proxy).detach().cpu().numpy()
        return self.env.plot_reward_samples(
            state, scores, "Context Points across Fidelities"
        )


# class GaussianProcessMultiFidelityMES():
#     def __init__()
#         self.load
