from gflownet.envs.amp import AMP as GflowNetAMP
import torch
from torchtyping import TensorType
from typing import List, Tuple
from .base import GFlowNetEnv
from clamp_common_eval.defaults import get_default_data_splits
from sklearn.model_selection import GroupKFold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class AMP(GFlowNetEnv, GflowNetAMP):
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
            oracle_states = self.statetorch2oracle(states)
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

    # def initialize_dataset(self, config):
    #     train_df = pd.read_csv(config.train.path)
    #     train_states = train_df["samples"]
    #     train_states = train_states.values.tolist()
    #     train_states = torch.tensor(states)
    #     # such that nucleotide count lies in 0 -3
    #     scores = train_df["energies"]
    #     return states, scores

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
            states_oracle_input = states.clone()
            oracle_states = self.statetorch2oracle(states_oracle_input)
            scores = self.oracle(oracle_states)

        if resume == False:
            return states, scores
        else:
            return train_states, train_scores, test_states, test_scores

    def load_dataset(self, split="D1", nfold=5):
        # TODO: rename to make_dataset()?
        source = get_default_data_splits(setting="Target")
        rng = np.random.RandomState()
        # returns a dictionary with two keys 'AMP' and 'nonAMP' and values as lists
        data = source.sample(split, -1)
        if split == "D1":
            groups = np.array(source.d1_pos.group)
        if split == "D2":
            groups = np.array(source.d2_pos.group)
        if split == "D":
            groups = np.concatenate(
                (np.array(source.d1_pos.group), np.array(source.d2_pos.group))
            )

        n_pos, n_neg = len(data["AMP"]), len(data["nonAMP"])
        pos_train, pos_test = next(
            GroupKFold(nfold).split(np.arange(n_pos), groups=groups)
        )
        neg_train, neg_test = next(
            GroupKFold(nfold).split(
                np.arange(n_neg), groups=rng.randint(0, nfold, n_neg)
            )
        )

        pos_train = [data["AMP"][i] for i in pos_train]
        neg_train = [data["nonAMP"][i] for i in neg_train]
        pos_test = [data["AMP"][i] for i in pos_test]
        neg_test = [data["nonAMP"][i] for i in neg_test]
        train = pos_train + neg_train
        test = pos_test + neg_test

        train = [sample for sample in train if len(sample) < self.max_seq_length]
        test = [sample for sample in test if len(sample) < self.max_seq_length]

        return train, test

    def get_random_terminating_states(self, n_samples, **kwargs):
        states = torch.randint((5 * n_samples, self.max_seq_length), 0, 19)
        # select unique states
        states = torch.unique(states, dim=0)[:n_samples]
        list_of_states = list(states)
        return list_of_states
