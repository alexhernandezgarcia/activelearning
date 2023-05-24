import torch
from torchtyping import TensorType
from typing import List, Tuple

from gflownet.envs.sequence import Sequence as GflowNetSequence
from utils.selfies import SELFIES_VOCAB_SMALL, SELFIES_VOCAB_LARGE
from .base import GFlowNetEnv
import pandas as pd
import selfies as sf
import matplotlib.pyplot as plt

# from rdkit import Chem, AllChem, DataStructs


class GFlowNetMolSelfies(GflowNetSequence):
    """
    Molecular environment as a SELFIES sequence
    """

    def __init__(
        self,
        selfies_vocab="small",
        **kwargs,
    ):

        special_tokens = ["[nop]", "[EOS]"]
        # "[SEP]", "[UNK]", "[MASK]"

        if selfies_vocab == "small":
            selfies_vocab = SELFIES_VOCAB_SMALL
        elif selfies_vocab == "large":
            selfies_vocab = SELFIES_VOCAB_LARGE
        else:
            raise NotImplementedError
        self.vocab = selfies_vocab + special_tokens
        super().__init__(
            **kwargs,
            special_tokens=special_tokens,
        )
        # pad_to_len = self.
        # max(sf.len_selfies(s) for s in selfies)  # 5
        # self.vocab = list(sorted(self.vocab))
        # self.symbol_to_idx = {s: i for i, s in enumerate(self.vocab)}


class MolSelfies(GFlowNetEnv, GFlowNetMolSelfies):
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

    def readable2state(self, readable):
        state = sf.selfies_to_encoding(
            selfies=readable,
            vocab_stoi=self.lookup,
            pad_to_len=self.max_seq_length,
            enc_type="label",
        )
        return torch.tensor(state, device=self.device)

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

    def initialize_dataset(self, config, n_samples, resume, **kwargs):
        train_scores = torch.tensor([], device=self.device)
        test_scores = torch.tensor([], device=self.device)
        train_states = torch.tensor([], device=self.device)
        test_states = torch.tensor([], device=self.device)

        if config.oracle_dataset is not None:
            if config.oracle_dataset.train is not None:
                train_df = pd.read_csv(config.oracle_dataset.train.path)
                train_states = train_df["samples"].values.tolist()
                train_states = [
                    torch.tensor(self.readable2state(sample)) for sample in train_states
                ]
                train_states = torch.vstack(train_states).to(self.device)
                if config.oracle_dataset.train.get_scores == False:
                    train_scores = train_df["energies"].values.tolist()
                    train_scores = torch.tensor(train_scores)

            if config.oracle_dataset.test is not None:
                test_df = pd.read_csv(config.oracle_dataset.test.path)
                test_states = test_df["samples"].values.tolist()
                test_states = [
                    torch.tensor(self.readable2state(sample)) for sample in test_states
                ]
                test_states = torch.vstack(test_states).to(self.device)
                if config.oracle_dataset.test.get_scores == False:
                    test_scores = test_df["energies"].values.tolist()
                    test_scores = torch.tensor(test_scores)

        if len(train_states) == 0 and len(test_states) == 0:
            states = self.get_random_terminating_states(n_samples)
        else:
            states = torch.cat((train_states, test_states))
            scores = torch.cat((train_scores, test_scores))
            if len(scores) < 0:
                idxNAN = torch.isnan(scores)
                states = states[~idxNAN]
                scores = scores[~idxNAN]

        if len(scores) == 0 or len(scores) != len(states):
            train_states_oracle_input = train_states.clone()
            train_oracle_states = self.statetorch2oracle(train_states_oracle_input)
            train_scores = self.oracle(train_oracle_states)
            idxNAN = torch.isnan(train_scores).to(train_scores.device)
            train_states = train_states[~idxNAN]
            train_scores = train_scores[~idxNAN]

            test_states_oracle_input = test_states.clone()
            test_oracle_states = self.statetorch2oracle(test_states_oracle_input)
            test_scores = self.oracle(test_oracle_states)
            idxNAN = torch.isnan(test_scores).to(test_scores.device)
            test_states = test_states[~idxNAN]
            test_scores = test_scores[~idxNAN]

            scores = torch.cat((train_scores, test_scores))

        if resume == False:
            return states, scores
        else:
            return train_states, train_scores, test_states, test_scores
        
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
