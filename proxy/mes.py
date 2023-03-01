from botorch.acquisition.max_value_entropy_search import (
    qMultiFidelityLowerBoundMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
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
from mpl_toolkits.axes_grid1 import make_axes_locatable


CLAMP_LB = 1.0e-8
from botorch.models.utils import check_no_nans
from .botorch_models import FidelityCostModel
from abc import abstractmethod
import matplotlib.pyplot as plt


class MultiFidelity(Proxy):
    def __init__(self, env, cost_type, fixed_cost, logger, **kwargs):
        super().__init__(**kwargs)
        self.env = env
        self.cost_type = cost_type
        self.fixed_cost = fixed_cost
        self.logger = logger
        self.n_fid = len(self.env.fidelity_costs)
        self.candidate_set = self.load_candidate_set()

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
                float_precision=self.float,
            )
        elif self.cost_type == "botorch":
            cost_model = AffineFidelityCostModel(
                fidelity_weights={-1: self.env.oracle[self.n_fid - 1].fid},
                fixed_cost=self.fixed_cost,
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
        if isinstance(states, tuple):
            states = torch.cat((states[0], states[1]), dim=1)
        if isinstance(states, np.ndarray):
            # for the case when buffer test states are input
            states = torch.tensor(states, dtype=self.float, device=self.device)
        # add extra dimension for q
        states = states.unsqueeze(-2)
        mes = self.acqf(states)
        return mes

    def plot_context_points(self, **kwargs):
        pass


class MES(MultiFidelity):
    def __init__(self, gibbon=True, **kwargs):
        super().__init__(**kwargs)
        cost_aware_utility = self.get_cost_utility()
        model = self.load_model()
        if gibbon == True:
            self.acqf = qMultiFidelityLowerBoundMaxValueEntropy(
                model,
                candidate_set=self.candidate_set,
                project=self.project,
                cost_aware_utility=cost_aware_utility,
            )
        else:
            self.acqf = qMultiFidelityMaxValueEntropy(
                model,
                candidate_set=self.candidate_set,
                project=self.project,
                cost_aware_utility=cost_aware_utility,
                gibbon=False,
            )


class ProxyMultiFidelityMES(MES):
    def __init__(self, regressor, num_dropout_samples, **kwargs):
        self.regressor = regressor
        self.num_dropout_samples = num_dropout_samples
        if not self.regressor.load_model():
            raise FileNotFoundError
        super().__init__(**kwargs)

    def load_model(self):
        model = MultiFidelityProxyModel(
            self.regressor, self.num_dropout_samples, self.n_fid, device=self.device
        )
        return model

    def project(self, states):
        input_dim = states.ndim
        max_fid = torch.zeros((states.shape[0], self.n_fid), device=self.device)
        max_fid[:, -1] = 1
        if input_dim == 3:
            states = states[:, :, : -self.n_fid]
            max_fid = max_fid.unsqueeze(1)
            states = torch.cat([states, max_fid], dim=2)
        elif input_dim == 2:
            states = states[:, : -self.n_fid]
            states = torch.cat([states, max_fid], dim=1)
        return states

    def load_candidate_set(self):
        path = self.logger.data_path.parent / Path("data_train.csv")
        data = pd.read_csv(path, index_col=0)
        samples = data["samples"]
        state = [self.env.readable2state(sample) for sample in samples]
        # states = pad_sequence(
        #         states,
        #         batch_first=True,
        #         padding_value=self.env.env.invalid_state_element,
        #     )
        state_proxy = self.env.statebatch2proxy(state)
        if isinstance(state_proxy, torch.Tensor):
            return state_proxy.to(self.device)
        else:
            state_proxy = torch.tensor(
                state_proxy, device=self.device, dtype=self.float
            )
        return state_proxy


class OracleMultiFidelityMES(MES):
    def __init__(self, data_path, oracle, **kwargs):
        self.data_path = data_path
        self.oracle = oracle
        super().__init__(**kwargs)
        fig = self.plot_acquisition_rewards()
        self.logger.log_figure("acquisition_rewards", fig, use_context=True)

    def load_candidate_set(self):
        path = self.data_path
        data = pd.read_csv(path, index_col=0)
        samples = data["samples"]
        # samples have states and fidelity
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

    def plot_acquisition_rewards(self, **kwargs):
        states = torch.tensor(
            self.env.env.get_all_terminating_states(), dtype=self.float
        ).to(self.device)
        n_states = states.shape[0]
        fidelities = torch.zeros((len(states) * self.n_fid, 1), dtype=self.float).to(
            self.device
        )
        for i in range(self.n_fid):
            fidelities[i * len(states) : (i + 1) * len(states), 0] = self.env.oracle[
                i
            ].fid
        states = states.repeat(self.n_fid, 1)
        state_fid = torch.cat([states, fidelities], dim=1)
        states_oracle, fid = self.env.statetorch2oracle(state_fid)
        # Specific to grid as the states are transformed to oracle space on feeding to MES
        if isinstance(states_oracle, torch.Tensor):
            states_fid_oracle = torch.cat([states_oracle, fid], dim=1)
        states = states[:n_states]

        scores = self(states_fid_oracle).detach().cpu().numpy()
        width = (self.n_fid) * 5
        fig, axs = plt.subplots(1, self.n_fid, figsize=(width, 5))
        for fid in range(0, self.n_fid):
            index = states.long().detach().cpu().numpy()
            grid_scores = np.zeros((self.env.env.length, self.env.env.length))
            grid_scores[index[:, 0], index[:, 1]] = scores[
                fid * len(states) : (fid + 1) * len(states)
            ]
            axs[fid].set_xticks(np.arange(self.env.env.length))
            axs[fid].set_yticks(np.arange(self.env.env.length))
            axs[fid].imshow(grid_scores)
            axs[fid].set_title(
                "Oracle-Mes Reward with fid {}".format(self.env.oracle[fid].fid)
            )
            im = axs[fid].imshow(grid_scores)
            divider = make_axes_locatable(axs[fid])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()
        plt.tight_layout()
        plt.close()
        return fig


class DeepKernelMultiFidelityMES(MES):
    def __init__(self, regressor, **kwargs):
        self.regressor = regressor
        super().__init__(**kwargs)

    def load_model(self):
        return self.regressor.surrogate

    def load_candidate_set(self):
        path = self.logger.data_path.parent / Path("data_train.csv")
        data = pd.read_csv(path, index_col=0)
        samples = data["samples"]
        state = [self.env.readable2state(sample) for sample in samples]
        state_proxy = self.env.statebatch2proxy(state)  # [: self.regressor.n_samples]
        fid_proxy = state_proxy[..., -1]
        state_proxy = state_proxy[..., :-1]
        (
            state_tok_features,
            state_mask,
        ) = self.regressor.language_model.get_token_features(state_proxy)
        _, pooled_state_proxy_features = self.regressor.language_model.pool_features(
            state_tok_features, state_mask
        )
        candidate_set = torch.cat(
            [pooled_state_proxy_features, fid_proxy.unsqueeze(-1)], dim=1
        )
        return candidate_set

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

    def __call__(self, states):
        assert torch.eq(states.to(torch.long), states).all()
        inputs = states.clone().to(torch.long)
        fid = inputs[..., -1]
        inputs = inputs[..., :-1]
        (
            input_tok_features,
            input_mask,
        ) = self.regressor.language_model.get_token_features(inputs)
        _, pooled_features = self.regressor.language_model.pool_features(
            input_tok_features, input_mask
        )
        inputs = pooled_features
        inputs = torch.cat([inputs, fid.unsqueeze(-1)], dim=1)
        inputs = inputs.unsqueeze(-2)
        acq_values = self.acqf(inputs)
        return acq_values


class GaussianProcessMultiFidelityMES(MES):
    def __init__(self, regressor, **kwargs):
        self.regressor = regressor
        super().__init__(**kwargs)
        fig = self.plot_acquisition_rewards()
        if fig is not None:
            self.logger.log_figure("acquisition_rewards", fig, use_context=True)

    def load_model(self):
        return self.regressor.model

    def load_candidate_set(self):
        path = self.logger.data_path.parent / Path("data_train.csv")
        data = pd.read_csv(path, index_col=0)
        samples = data["samples"]
        state = [self.env.readable2state(sample) for sample in samples]
        state_proxy = self.env.statebatch2proxy(state)  # [: self.regressor.n_samples]
        if isinstance(state_proxy, torch.Tensor):
            return state_proxy.to(self.device)
        return torch.FloatTensor(state_proxy).to(self.device)

    def project(self, states):
        input_dim = states.ndim
        max_fid = torch.ones((states.shape[0], 1), device=self.device).long() * (
            self.env.oracle[self.n_fid - 1].fid
        )
        if input_dim == 3:
            states = states[:, :, :-1]
            max_fid = max_fid.unsqueeze(1)
            states = torch.cat([states, max_fid], dim=2)
        elif input_dim == 2:
            states = states[:, :-1]
            states = torch.cat([states, max_fid], dim=1)
        return states

    def plot_acquisition_rewards(self, **kwargs):
        if hasattr(self.env.env, "get_all_terminating_states") == False:
            return None
        states = torch.tensor(
            self.env.env.get_all_terminating_states(), dtype=self.float
        ).to(self.device)
        n_states = states.shape[0]
        fidelities = torch.zeros((len(states) * self.n_fid, 1), dtype=self.float).to(
            self.device
        )
        for i in range(self.n_fid):
            fidelities[i * len(states) : (i + 1) * len(states), 0] = self.env.oracle[
                i
            ].fid
        states = states.repeat(self.n_fid, 1)
        state_fid = torch.cat([states, fidelities], dim=1)
        states_oracle, fid = self.env.statetorch2oracle(state_fid)
        # Specific to grid as the states are transformed to oracle space on feeding to MES
        if isinstance(states_oracle, torch.Tensor):
            states_fid_oracle = torch.cat([states_oracle, fid], dim=1)
        states = states[:n_states]

        scores = self(states_fid_oracle).detach().cpu().numpy()
        width = (self.n_fid) * 5
        fig, axs = plt.subplots(1, self.n_fid, figsize=(width, 5))
        if self.env.rescale != 1:
            step = int(self.env.env.length / self.env.rescale)
        else:
            step = 1
        for fid in range(0, self.n_fid):
            index = states.long().detach().cpu().numpy()
            grid_scores = np.zeros((self.env.env.length, self.env.env.length))
            grid_scores[index[:, 0], index[:, 1]] = scores[
                fid * len(states) : (fid + 1) * len(states)
            ]
            axs[fid].set_xticks(
                np.arange(
                    start=0,
                    stop=self.env.env.length,
                    step=step,
                )
            )
            axs[fid].set_yticks(
                np.arange(
                    start=0,
                    stop=self.env.env.length,
                    step=step,
                )
            )
            axs[fid].imshow(grid_scores)
            axs[fid].set_title(
                "GP-Mes Reward with fid {}".format(self.env.oracle[fid].fid)
            )
            im = axs[fid].imshow(grid_scores)
            divider = make_axes_locatable(axs[fid])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()
        plt.tight_layout()
        plt.close()
        return fig
