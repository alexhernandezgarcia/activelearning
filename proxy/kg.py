from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from .mes import MultiFidelity
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import numpy as np
import pandas as pd
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.analytic import PosteriorMean
from botorch.optim.optimize import optimize_acqf
from botorch.optim.optimize import optimize_acqf_mixed

class GaussianProcessKnowledgeGradient(MultiFidelity):
    def __init__(self, regressor, **kwargs):
        self.regressor = regressor
        super().__init__(**kwargs)
        cost_aware_utility = self.get_cost_utility()
        model = self.load_model()
        dim = 3
        curr_val_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(model),
            d=dim,
            columns=[2],
            values=[1],
            )
        bounds =  torch.tensor([[0.0] * dim, [1.0] * dim], dtype=self.float, device=self.device)
        _, current_value = optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=bounds[:, :-1],
            q=1,
            num_restarts=2,
            raw_samples= 4,
        )
        # We need to maximise the function
        current_value = -min(self.candidate_set)
        self.acqf = qMultiFidelityKnowledgeGradient(
            model=model,
            current_value=current_value,
            cost_aware_utility=cost_aware_utility,
            project=self.project,
        )
        fig = self.plot_acquisition_rewards()
        self.logger.log_figure("acquisition_rewards", fig, use_context=True)

    def load_model(self):
        return self.regressor.model

    def load_candidate_set(self):
        path = self.logger.data_path.parent / Path("data_train.csv")
        data = pd.read_csv(path, index_col=0)
        # samples = data["samples"]
        energies = data["energies"]
        # state = [self.env.readable2state(sample) for sample in samples]
        # state_proxy = self.env.statebatch2proxy(state)[: self.regressor.n_samples]
        # if isinstance(energies, torch.Tensor):
        # return energies.to(self.device)
        return torch.tensor(energies, device=self.device, dtype=self.float)

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
                "GP-KG Reward with fid {}".format(self.env.oracle[fid].fid)
            )
            im = axs[fid].imshow(grid_scores)
            divider = make_axes_locatable(axs[fid])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()
        plt.tight_layout()
        plt.close()
        return fig
