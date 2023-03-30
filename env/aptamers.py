from gflownet.envs.aptamers import Aptamers as GflowNetAptamers
import torch
from torchtyping import TensorType
from typing import List, Tuple
from .base import GFlowNetEnv
from clamp_common_eval.defaults import get_default_data_splits
from sklearn.model_selection import GroupKFold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def split_str(s):
    return [int(ch) for ch in s]


class Aptamers(GFlowNetEnv, GflowNetAptamers):
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs,
        )
        if self.proxy_state_format == "ohe":
            self.statebatch2proxy = self.statebatch2policy
            self.statetorch2proxy = self.statetorch2policy
        elif self.proxy_state_format == "oracle":
            self.statebatch2proxy = self.statebatch2oracle
            self.statetorch2proxy = self.statetorch2oracle
        elif self.proxy_state_format == "state":
            self.statebatch2proxy = self.statebatch2state
            self.statetorch2proxy = self.statetorch2state
        else:
            raise ValueError(
                "Invalid proxy_state_format: {}".format(self.proxy_state_format)
            )
        self.tokenizer = None

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def unpad_function(self, states_term):
        states_tensor = torch.tensor(states_term)
        state_XX = []
        for state in states_tensor:
            state = (
                state[: torch.where(state == self.padding_idx)[0][0]]
                if state[-1] == self.padding_idx
                else state
            )
            state_XX.append(state)
        return state_XX

    def statebatch2state(
        self, states: List[TensorType["1", "state_dim"]]
    ) -> TensorType["batch", "state_dim"]:
        if self.tokenizer is not None:
            states = torch.vstack(states)
            states = self.tokenizer.transform(states)
        return states.to(self.device)

    def statetorch2state(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "state_dim"]:
        if self.tokenizer is not None:
            states = self.tokenizer.transform(states)
        return states.to(self.device)

    def plot_reward_distribution(
        self, states=None, scores=None, ax=None, title=None, oracle=None, **kwargs
    ):
        if ax is None:
            fig, ax = plt.subplots()
            standalone = True
        else:
            standalone = False
        if oracle is None:
            oracle = self.oracle
        if title == None:
            title = "Rewards of Sampled States"
        if scores is None:
            oracle_states = self.statebatch2oracle(states)
            scores = oracle(oracle_states)
        if isinstance(scores, TensorType):
            scores = scores.cpu().detach().numpy()
        ax.hist(scores)
        ax.set_title(title)
        ax.set_ylabel("Number of Samples")
        ax.set_xlabel("Energy")
        plt.show()
        if standalone == True:
            plt.tight_layout()
            plt.close()
        return ax

    def initialize_dataset(self, config, n_samples, **kwargs):
        train_df = pd.read_csv(config.oracle_dataset.train.path)
        states = train_df["indices"].apply(split_str)
        states = states.values.tolist()
        states = torch.tensor(states)
        # such that nucleotide count lies in 0 -3
        states = states - 1
        scores = train_df["energies"].values.tolist()
        scores = torch.tensor(scores)
        return states, scores

    def get_random_terminating_states(self, n_samples, **kwargs):
        states = torch.randint(low=0, high=4, size=(5 * n_samples, self.max_seq_length))
        # select unique states
        states = torch.unique(states, dim=0)[:n_samples]
        list_of_states = list(states)
        return list_of_states
