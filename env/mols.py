import torch
from torchtyping import TensorType
from typing import List, Tuple

from gflownet.envs.sequence import Sequence as GflowNetSequence
from utils.selfies import SELFIES_VOCAB_SMALL, SELFIES_VOCAB_LARGE
from .base import GFlowNetEnv


class GFlowNetMolSelfies(GflowNetSequence):
    """
    Molecular environment as a SELFIES sequence
    """

    def __init__(
        self,
        selfies_vocab="small",
        **kwargs,
    ):
        special_tokens = ["[nop]", "[CLS]", "[SEP]", "[UNK]", "[MASK]", "[PAD]"]
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
