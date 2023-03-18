from gflownet.envs.amp import AMP as GflowNetAMP
import torch
from torchtyping import TensorType
from typing import List, Tuple
from .base import GFlowNetEnv
from clamp_common_eval.defaults import get_default_data_splits
from sklearn.model_selection import GroupKFold
import numpy as np


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
