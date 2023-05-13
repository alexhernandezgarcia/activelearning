from gflownet.envs.aptamers import Aptamers as GflowNetAptamers
import torch
from torchtyping import TensorType
from typing import List, Tuple
from .base import GFlowNetEnv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import random


def split_str(s):
    return [int(ch) for ch in s]


class Aptamers(GFlowNetEnv, GflowNetAptamers):
    def __init__(self, max_init_steps, **kwargs):
        self.max_init_steps = max_init_steps
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
        # self.tokenizer = None

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def reset(self, env_id=None):
        self.state = (
            self.source.clone()
        )  
        init_steps = random.randint(0, self.max_init_steps)
        random_state = torch.randint(low=0, high=4, size=(1, init_steps))
        self.state[0:init_steps] = random_state
        self.n_actions = 0
        self.done = False
        self.id = env_id
        return self

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
            if states is None or len(states) == 0:
                return None
            oracle_states = self.statebatch2oracle(states)
            scores = oracle(oracle_states)
        if isinstance(scores, TensorType):
            scores = scores.cpu().detach().numpy()
        ax.hist(scores)
        ax.set_title(title)
        ax.set_ylabel("Number of Samples")
        ax.set_xlabel("Energy")
        if "MES" in title:
            ax.set_xbound(-0.0, 0.01)
            # ax.set_xticks(np.arange(start = 0.0, stop=0.1, step=1e-2))
        plt.show()
        if standalone == True:
            plt.tight_layout()
            plt.close()
        return ax

    # def energy_vs_reward(self, energies, rewards):
    #     # Plot a scatter plot of energy vs reward
    #     fig, ax = plt.subplots()
    #     ax.scatter(energies, rewards)
    #     ax.set_ybound(0.0, 0.01)
    #     ax.set_xlabel("Energy")
    #     ax.set_ylabel("Reward")
    #     plt.show()
    #     plt.close()
    #     return ax

    def initialize_dataset(self, config, n_samples, resume, **kwargs):
        train_scores = torch.tensor([])
        test_scores = torch.tensor([])
        train_states = torch.tensor([])
        test_states = torch.tensor([])

        if config.oracle_dataset is not None:
            if config.oracle_dataset.train is not None:
                train_df = pd.read_csv(config.oracle_dataset.train.path)
                train_states = train_df["samples"].values.tolist()
                train_states = [
                    torch.tensor(self.readable2state(sample)) for sample in train_states
                ]
                train_states = torch.vstack(train_states)
                if config.oracle_dataset.train.get_scores == False:
                    train_scores = train_df["energies"].values.tolist()
                    train_scores = torch.tensor(train_scores)

            if config.oracle_dataset.test is not None:
                test_df = pd.read_csv(config.oracle_dataset.test.path)
                test_states = test_df["samples"].values.tolist()
                test_states = [
                    torch.tensor(self.readable2state(sample)) for sample in test_states
                ]
                test_states = torch.vstack(test_states)
                if config.oracle_dataset.test.get_scores == False:
                    test_scores = test_df["energies"].values.tolist()
                    test_scores = torch.tensor(test_scores)

        if len(train_states) == 0 and len(test_states) == 0:
            states = self.get_random_terminating_states(n_samples)
        else:
            states = torch.cat((train_states, test_states))
            scores = torch.cat((train_scores, test_scores))

        if len(scores) == 0 or len(scores) != len(states):
            train_states_oracle_input = train_states.clone()
            train_oracle_states = self.statetorch2oracle(train_states_oracle_input)
            train_scores = self.oracle(train_oracle_states)

            test_states_oracle_input = test_states.clone()
            test_oracle_states = self.statetorch2oracle(test_states_oracle_input)
            test_scores = self.oracle(test_oracle_states)

            scores = torch.cat((train_scores, test_scores))

        if resume == False:
            return states, scores
        else:
            return train_states, train_scores, test_states, test_scores

    # def load_test_dataset(self, logger):
    #     # path = logger.data_path.parent / Path("data_test.csv")
    #     print("Loading UNIFORM Test Dataset")
    #     path = Path("/home/mila/n/nikita.saxena/activelearning/storage/dna/length30/test_2000_FINAL.csv")
    #     dataset = pd.read_csv(path, index_col=0)
    #     samples = dataset["samples"]
    #     scores = dataset["energies"]
    #     states = [self.readable2state(sample) for sample in samples]
    #     states = torch.stack(states)
    #     return states, scores

    def write_samples_to_file(self, samples, path):
        samples = [self.state2readable(state) for state in samples]
        df = pd.DataFrame(samples, columns=["samples"])
        df.to_csv(path)

    # def initialize_dataset(self, config, n_samples, resume, **kwargs):
    #     train_states = torch.tensor([])
    #     train_scores = torch.tensor([])
    #     test_states = torch.tensor([])
    #     test_scores = torch.tensor([])
    #     if config.oracle_dataset is not None:
    #         if config.oracle_dataset.train is not None:
    #             train_df = pd.read_csv(config.oracle_dataset.train.path)
    #             train_states = train_df["indices"].apply(split_str)
    #             train_states = train_states.values.tolist()
    #             train_states = torch.tensor(train_states)
    #             # such that nucleotide count lies in 0 -3
    #             train_states = train_states - 1
    #             train_scores = train_df["energies"].values.tolist()
    #             train_scores = torch.tensor(train_scores)
    #         if config.oracle_dataset.test is not None:
    #             test_df = pd.read_csv(config.oracle_dataset.test.path)
    #             test_states = test_df["indices"].apply(split_str)
    #             test_states = test_states.values.tolist()
    #             test_states = torch.tensor(test_states)
    #             # such that nucleotide count lies in 0 -3
    #             test_states = test_states - 1
    #             test_scores = test_df["energies"].values.tolist()
    #             test_scores = torch.tensor(test_scores)
    #     if test_states.shape[0] == 0:
    #         states = train_states
    #         scores = train_scores
    #     else:
    #         states = torch.vstack([train_states, test_states])
    #         scores = torch.cat([train_scores, test_scores])
    #     # states = train_states + test_states
    #     # scores = train_scores + test_scores
    #     if resume == False:
    #         return states, scores
    #     else:
    #         return train_states, train_scores, test_states, test_scores

    def get_random_terminating_states(self, n_samples, **kwargs):
        states = torch.randint(low=0, high=4, size=(5 * n_samples, self.max_seq_length))
        # select unique states
        states = torch.unique(states, dim=0)[:n_samples]
        list_of_states = list(states)
        return list_of_states
