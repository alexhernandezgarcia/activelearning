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
        bounds = torch.tensor(
            [[0.0] * dim, [1.0] * dim], dtype=self.float, device=self.device
        )
        _, current_value = optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=bounds[:, :-1],
            q=1,
            num_restarts=1,
            raw_samples=4,
            options={"batch_limit": 10, "maxiter": 200},
        )
        # We need to maximise the function
        # current_value = -min(self.candidate_set)
        self.acqf = qMultiFidelityKnowledgeGradient(
            model=model,
            current_value=current_value,
            cost_aware_utility=cost_aware_utility,
            project=self.project,
            num_fantasies=1,
        )
        fig = self.plot_acquisition_rewards()
        self.logger.log_figure("acquisition_rewards", fig, use_context=True)

    def load_model(self):
        return self.regressor.model

    def __call__(self, states):
        states = states.unsqueeze(-2)
        states = torch.cat([states, states], dim=-2)
        kg = self.acqf(states)
        return kg

    def load_candidate_set(self):
        path = self.logger.data_path.parent / Path("data_train.csv")
        data = pd.read_csv(path, index_col=0)
        energies = data["energies"]
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
                "GP-Mes AcqF Val with fid {} and cost {}".format(
                    self.env.oracle[fid].fid, self.env.oracle[fid].cost
                )
            )
            im = axs[fid].imshow(grid_scores)
            divider = make_axes_locatable(axs[fid])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
            plt.show()
        plt.tight_layout()
        plt.close()
        return fig
