from abc import abstractmethod
from typing import List, Tuple
import numpy.typing as npt
from torchtyping import TensorType
from gflownet.envs.base import GFlowNetEnv
import numpy as np
import torch


class MultiFidelityEnvWrapper(GFlowNetEnv):
    """
    Multi-fidelity environment for GFlowNet.
    Assumes same data transformation required for all oracles.
    Does not require the different oracles as scoring is performed by GFN not env
    """

    def __init__(self, env, n_fid, oracle, is_oracle_proxy=False, **kwargs):
        self.env = env
        self.n_fid = n_fid
        self.fid = self.env.eos + 1
        # Member variables of env are also member variables of MFENV
        # function variables of mfenv are not updated with that of env
        # TODO: can we skip this?
        vars(self).update(vars(self.env))
        self.action_space = self.get_actions_space()
        self.fixed_policy_output = self.get_fixed_policy_output()
        self.policy_input_dim = len(self.state2policy())
        self.random_policy_output = self.fixed_policy_output
        if is_oracle_proxy:
            # Assumes that all oracles required the same kind of transformed dtata
            self.statebatch2proxy = self.statebatch2oracle
            self.statetorch2proxy = self.statetorch2oracle
            # self.proxy = self.call_oracle_per_fidelity
        # self.state2oracle = self.env.state2oracle
        self.oracle = oracle
        self.reset()

    def copy(self):
        env_copy = self.env.copy()
        class_var = self.__dict__.copy()
        class_var.pop("env", None)
        return MultiFidelityEnvWrapper(**class_var, env=env_copy)

    def call_oracle_per_fidelity(self, state: TensorType["batch", "state_dim"]):
        fidelities = state[:, -1].long()
        scores = torch.zeros((state.shape[0]), dtype=self.float, device=self.device)
        state_oracle = state[:, :-1]
        for fid in range(self.n_fid):
            idx_fid = torch.where(fidelities == fid)[0]
            scores[idx_fid] = self.oracle[fid](state_oracle[idx_fid])
        return scores

    def get_actions_space(self):
        actions = self.env.get_actions_space()
        for fid_index in range(self.n_fid):
            actions = actions + [(self.fid, fid_index)]
        self.action_max_length = max(len(action) for action in actions)
        # assumes all actions in the sf-env are of the same length
        # TODO: remove self.action_max_length if we can make the above assumption
        self.action_pad_length = self.action_max_length - len(actions[0])
        if self.action_pad_length > 0:
            actions = [
                tuple(list(action) + [0] * (self.action_max_length - len(action)))
                for action in actions
            ]
        return actions

    def reset(self, env_id=None, fid: int = -1):
        # Fid in the state should range from 0 to self.n_fid -1 for easy one hot encoding
        # Offline Trajectory: might want to reset the env with a specific fid
        # TODO: does member variable env need to be updated?
        # env is a member variable of MFENV because we need the functions
        self.env = self.env.reset(env_id)
        # If fid is required as a member variable
        # then it should not be named self.fid as self.fid represents fidelity action
        self.state = self.env.state + [fid]
        # Update the variables that were reset and are required in mf
        self.fid_done = self.state[-1] != -1
        self.done = self.env.done and self.fid_done
        self.id = self.env.id
        return self

    def get_mask_invalid_actions_forward(self, state=None, done=None):
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True for _ in range(len(self.action_space))]
        mask = self.env.get_mask_invalid_actions_forward(state[:-1])
        mask = mask + [self.fid_done for _ in range(self.n_fid)]
        return mask

    def state2policy(self, state: List = None):
        if state is None:
            state = self.state.copy()
        state_policy = self.env.state2policy(state[:-1])
        fid_policy = np.zeros((self.n_fid), dtype=np.float32)
        if state[-1] != -1:
            fid_policy[state[-1]] = 1
        state_fid_policy = np.concatenate((state_policy, fid_policy), axis=0)
        return state_fid_policy

    def statebatch2policy(self, states: List[List]) -> npt.NDArray[np.float32]:
        if not states:
            return np.array(states)
        state_policy, fid_list = zip(*[(state[:-1], state[-1]) for state in states])
        state_policy = self.env.statebatch2policy(state_policy)
        fid_policy = np.zeros((len(states), self.n_fid), dtype=np.float32)
        fid_array = np.array(fid_list)
        index = np.where(fid_array != -1)[0]
        if index.size:
            fid_policy[index, fid_array[index]] = 1
        state_fid_policy = np.concatenate((state_policy, fid_policy), axis=1)
        return state_fid_policy

    def policy2state(self, state_policy: List) -> List:
        policy_no_fid = state_policy[:, : -self.n_fid]
        state_no_fid = self.env.policy2state(policy_no_fid)
        fid = np.argmax(state_policy[:, -self.n_fid :], axis=1)
        state = state_no_fid + fid.to_list()
        return state

    def state2readable(self, state=None):
        readable_state = self.env.state2readable(state[:-1])
        fid = str(state[-1])
        return readable_state + ";" + fid

    def readable2state(self, readable):
        fid = readable.split(";")[-1]
        readable_state = ";".join(readable.split(";")[:-1])
        state = self.env.readable2state(readable_state)
        state = state + [int(fid)]
        return state

    def get_parents(self, state=None, done=None, action=None):
        assert self.state[:-1] == self.env.state
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        parents_no_fid, actions = self.env.get_parents(state[:-1])
        if self.action_pad_length > 0:
            actions = [
                tuple(list(action) + [0] * (self.action_pad_length))
                for action in actions
            ]
        parents = [parent + [-1] for parent in parents_no_fid]
        if state[-1] != -1:
            fid = state[-1]
            parent = state[:-1] + [-1]
            actions.append(tuple([self.fid, fid] + [0] * (self.action_max_length - 2)))
            parents.append(parent)
        return parents, actions

    def step(self, action):
        assert self.state[:-1] == self.env.state
        if action[0] == self.fid:
            if self.fid_done == False:
                self.state[-1] = action[1]
                self.fid_done = True
            else:
                raise ValueError("Fidelity has been chosen")
            self.done = self.env.done and self.fid_done
            # TODO: action or eos in else
            # padded_action = tuple(list(action) + [0]*(self.action_pad_length-len(action)))
            return self.state, action, True
        else:
            fid = self.state[-1]
            env_action = action[: -self.action_pad_length]
            state, action, valid = self.env.step(env_action)
            self.state = state + [fid]
            self.done = self.env.done and self.fid_done
            padded_action = tuple(list(action) + [0] * (self.action_pad_length))
            return self.state, padded_action, valid

    def statetorch2policy(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "policy_input_dim"]:
        state_policy = states[:, :-1]
        state_policy = self.env.statetorch2policy(state_policy)
        fid_policy = torch.zeros(
            (states.shape[0], self.n_fid), dtype=torch.float32, device=self.device
        )
        index = torch.where(states[:, -1] != -1)[0]
        if index.size:
            fid_policy[index, states[index, -1].long()] = 1
        state_fid_policy = torch.cat((state_policy, fid_policy), dim=1)
        return state_fid_policy

    def statebatch2oracle(self, states: List[List]):
        states, fid_list = zip(*[(state[:-1], state[-1]) for state in states])
        # TODO: state_oracle is tensor in grid but list of strings in AMP. How to make this code compatible with both?
        state_oracle = self.env.statebatch2oracle(states)
        fid_torch = torch.Tensor(fid_list).long().to(self.device).unsqueeze(-1)
        # TODO: fix assertion -- runs into tuple error
        # assert torch.where(fid_torch == -1)[0].shape == 0
        state_fid = torch.concatenate((state_oracle, fid_torch), axis=1)
        return state_fid

    def statetorch2oracle(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "oracle_dim"]:
        state_oracle = states[:, :-1]
        state_oracle = self.env.statetorch2oracle(state_oracle)
        fid = states[:, -1].unsqueeze(-1)
        state_fid = torch.cat((state_oracle, fid), dim=1)
        return state_fid
