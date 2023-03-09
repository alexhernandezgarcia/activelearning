from botorch.acquisition.max_value_entropy_search import (
    qMultiFidelityLowerBoundMaxValueEntropy,
    qMultiFidelityMaxValueEntropy,
    qLowerBoundMaxValueEntropy,
    qMaxValueEntropy,
)
from gflownet.proxy.base import Proxy
from pathlib import Path
import pandas as pd
from .botorch_models import (
    MultifidelityOracleModel,
    MultiFidelityProxyModel,
    ProxyBotorchMES,
)
import torch
import numpy as np
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility

from gpytorch.functions import inv_quad
from mpl_toolkits.axes_grid1 import make_axes_locatable
from regressor.regressor import DropoutRegressor as SurrogateDropoutRegressor

CLAMP_LB = 1.0e-8
from botorch.models.utils import check_no_nans
from .botorch_models import FidelityCostModel
from abc import abstractmethod
import matplotlib.pyplot as plt
from regressor.dkl import DeepKernelRegressor


class SingleFidelityMES(Proxy):
    def __init__(self, regressor, logger, env, gibbon=True, **kwargs):
        super().__init__(**kwargs)
        self.regressor = regressor
        self.logger = logger
        self.env = env
        self.candidate_set = self.load_candidate_set()
        if isinstance(self.regressor, SurrogateDropoutRegressor):
            # TODO: Check if model weights are loaded.
            model = ProxyBotorchMES(self.regressor, self.num_dropout_samples)
        else:
            model = self.regressor.surrogate
            model.eval()
        if gibbon == True:
            self.acqf = qLowerBoundMaxValueEntropy(
                model,
                candidate_set=self.candidate_set,
            )
        else:
            self.acqf = qMaxValueEntropy(
                model,
                candidate_set=self.candidate_set,
            )

    def load_candidate_set(self):
        path = self.logger.data_path.parent / Path("data_train.csv")
        data = pd.read_csv(path, index_col=0)
        samples = data["samples"]
        state = [self.env.readable2state(sample) for sample in samples]
        state_proxy = self.env.statebatch2proxy(state)
        if isinstance(self.regressor, DeepKernelRegressor) == True:
            features = self.regressor.surrogate.get_features(state_proxy)
        return features


class GaussianProcessSingleFidelityMES(SingleFidelityMES):
    """
    Specifically for Deep Kernel Methods right now
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, states):
        if isinstance(self.regressor, DeepKernelRegressor) == True:
            features = self.regressor.surrogate.get_features(states)
        features = features.unsqueeze(-2)
        mes = self.acqf(features)
        return mes


class NeuralNetworkSingleFidelityMES(SingleFidelityMES):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, states):
        states = states.unsqueeze(-2)
        mes = self.acqf(states)
        return mes


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
        elif isinstance(state_proxy, tuple):
            state_proxy = torch.cat((state_proxy[0], state_proxy[1]), dim=1)
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
            self.env.get_all_terminating_states(), dtype=self.float
        ).to(self.device)
        n_states = self.env.env.length**2
        states_input_proxy = states.clone()
        states_proxy = self.env.statetorch2proxy(states_input_proxy)
        # states_proxy = torch.cat((states_proxy[0], states_proxy[1].unsqueeze(-1)), dim=1)
        scores = self(states_proxy).detach().cpu().numpy()
        states = states[:n_states]
        width = (self.n_fid) * 5
        fig, axs = plt.subplots(1, self.n_fid, figsize=(width, 5))
        if self.env.env.rescale != 1:
            step = int(self.env.env.length / self.env.env.rescale)
        else:
            step = 1
        for fid in range(0, self.n_fid):
            index = states.long().detach().cpu().numpy()
            grid_scores = np.zeros((self.env.env.length, self.env.env.length))
            grid_scores[index[:, 0], index[:, 1]] = scores[
                fid * len(states) : (fid + 1) * len(states)
            ]
            axs[fid].set_xticks(np.arange(start=0, stop=self.env.env.length, step=step))
            axs[fid].set_yticks(np.arange(start=0, stop=self.env.env.length, step=step))
            axs[fid].imshow(grid_scores)
            axs[fid].set_title(
                "Oracle-MES Reward with fid {}".format(self.env.oracle[fid].fid)
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
        if self.regressor.language_model.is_fid_param is False:
            state_proxy = state_proxy[..., :-1]
        if isinstance(self.regressor, DeepKernelRegressor) == True:
            if hasattr(self.regressor.language_model, "get_token_features"):
                (
                    input_tok_features,
                    input_mask,
                ) = self.regressor.language_model.get_token_features(state_proxy)
                _, candidate_features = self.regressor.language_model.pool_features(
                    input_tok_features, input_mask
                )
                candidate_features = torch.cat(
                    [candidate_features, fid_proxy.unsqueeze(-1)], dim=1
                )
            else:
                candidate_features = self.regressor.surrogate.get_features(state_proxy)
        # Get_features does the concatenation and returns
        # candidate_set = torch.cat(
        # [features, fid_proxy.unsqueeze(-1)], dim=1
        # )
        return candidate_features

    def project(self, states):
        input_dim = states.ndim
        # print(
        #     "User Defined Warning: In MES-project, fidelity is being replace by the INDEX (not oracle.fid) of the maximum fidelity."
        # )
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

    def __call__(self, inputs):
        # inputs = states.clone()
        fid = inputs[..., -1]
        if self.regressor.language_model.is_fid_param is False:
            inputs = inputs[..., :-1]
        if isinstance(self.regressor, DeepKernelRegressor) == True:
            if hasattr(self.regressor.language_model, "get_token_features"):
                (
                    input_tok_features,
                    input_mask,
                ) = self.regressor.language_model.get_token_features(inputs)
                _, features = self.regressor.language_model.pool_features(
                    input_tok_features, input_mask
                )
                features = torch.cat([features, fid.unsqueeze(-1)], dim=1)
            else:
                features = self.regressor.surrogate.get_features(inputs)

        # inputs = features
        features = features.unsqueeze(-2)
        acq_values = self.acqf(features)
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
        states = states.to(self.device)
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
        if self.env.n_dim > 2:
            return None
        if hasattr(self.env, "get_all_terminating_states") == False:
            return None
        states = torch.tensor(
            self.env.get_all_terminating_states(), dtype=self.float
        ).to(self.device)
        n_states = self.env.env.length**2
        states_input_proxy = states.clone()
        states_proxy = self.env.statetorch2proxy(states_input_proxy)
        scores = self(states_proxy).detach().cpu().numpy()
        states = states[:n_states]
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


class VariationalGPMultiFidelityMES(MES):
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
        state_proxy = self.env.statebatch2proxy(state)
        if isinstance(state_proxy, torch.Tensor):
            candidate_set = state_proxy.to(self.device)
        else:
            candidate_set = torch.tensor(
                state_proxy, device=self.device, dtype=self.float
            )
        return candidate_set

    def project(self, states):
        input_dim = states.ndim
        states = states.to(self.device)
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
