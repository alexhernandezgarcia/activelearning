from abc import abstractmethod
from typing import List, Tuple
import numpy.typing as npt
from torchtyping import TensorType
from gflownet.envs.base import GFlowNetEnv
import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools


class MultiFidelityEnvWrapper(GFlowNetEnv):
    """
    Multi-fidelity environment for GFlowNet.
    Assumes same data transformation required for all oracles.
    Does not require the different oracles as scoring is performed by GFN not env
    """

    def __init__(self, env, n_fid, oracle, proxy_state_format=False, **kwargs):
        # TODO: super init kwargs
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
        if proxy_state_format == "oracle":
            # Assumes that all oracles required the same kind of transformed dtata
            self.statebatch2proxy = self.statebatch2oracle
            self.statetorch2proxy = self.statetorch2oracle
        else:
            self.statebatch2proxy = self.statebatch2policy
            self.statetorch2proxy = self.statetorch2policy
            # self.proxy = self.call_oracle_per_fidelity
        # self.state2oracle = self.env.state2oracle
        self.oracle = oracle
        self.fidelity_costs = self.set_fidelity_costs()
        self.reset()

    def set_fidelity_costs(self):
        fidelity_costs = {}
        for idx in range(self.n_fid):
            fidelity_costs[float(idx)] = self.oracle[idx].cost
        return fidelity_costs

    def copy(self):
        env_copy = self.env.copy()
        class_var = self.__dict__.copy()
        class_var.pop("env", None)
        return MultiFidelityEnvWrapper(**class_var, env=env_copy)

    def call_oracle_per_fidelity(self, state_oracle, fidelities):
        scores = torch.zeros(
            (fidelities.shape[0]), dtype=self.float, device=self.device
        )
        for fid in range(self.n_fid):
            idx_fid = torch.where(fidelities == fid)[0]
            states = list(itertools.compress(state_oracle, idx_fid))
            scores[idx_fid] = self.oracle[fid](states)
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
        self.n_actions = 0
        return self

    def get_mask_invalid_actions_forward(self, state=None, done=None):
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            assert self.env.done == True
            return [True for _ in range(len(self.action_space))]
        mask = self.env.get_mask_invalid_actions_forward(state[:-1])
        mask = mask + [self.fid_done for _ in range(self.n_fid)]
        return mask

    def state2policy(self, state: List = None):
        if state is None:
            state = self.state.copy()
        state_policy = self.env.state2policy(state[:-1])
        fid_policy = np.zeros((self.n_fid), dtype=np.float32)
        if len(state) > 0 and state[-1] != -1:
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

    def statetorch2readable(self, state=None):
        readable_state = self.env.statetorch2readable(state[:-1])
        fid = str(state[-1].detach().cpu().numpy())
        # if isinstance(readable_state, list):
        #     # convert list to a string
        #     readable_state = str(readable_state)
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
        if state[-1] == -1:
            parents = [parent + [-1] for parent in parents_no_fid]
        elif state[-1] != -1:
            fid = state[-1]
            parents = [parent + [fid] for parent in parents_no_fid]
            parent = state[:-1] + [-1]
            actions.append(tuple([self.fid, fid] + [0] * (self.action_max_length - 2)))
            parents.append(parent)
        return parents, actions

    def step(self, action):
        assert self.state[:-1] == self.env.state
        assert self.done == (self.env.done and self.fid_done)
        if self.state[-1] == -1:
            assert self.n_actions == self.env.n_actions
        else:
            assert self.n_actions == self.env.n_actions + 1
        if self.done:
            raise ValueError("Action has been sampled despite environment being done")
            # return self.state, action, False
        if action[0] == self.fid:
            if self.fid_done == False:
                self.state[-1] = action[1]
                self.fid_done = True
                self.n_actions += 1
            else:
                raise ValueError("Fidelity has already been chosen.")
            self.done = self.env.done and self.fid_done
            # TODO: action or eos in else
            # padded_action = tuple(list(action) + [0]*(self.action_pad_length-len(action)))
            return self.state, action, True
        else:
            fid = self.state[-1]
            env_action = action[: -self.action_pad_length]
            state, action, valid = self.env.step(env_action)
            if valid:
                self.n_actions += 1
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
        # state_fid = torch.cat((state_oracle, fid), dim=1)
        return state_oracle, fid

    def true_density(self):
        if self._true_density is not None:
            return self._true_density
        # Calculate true density
        # TODO: fix when get_all_terminaiting states does not exist in env
        all_states = self.get_all_terminating_states()
        all_states = [
            state + [fid] for state in all_states for fid in range(self.n_fid)
        ]
        all_states = torch.FloatTensor(all_states).to(self.device)
        all_oracle = self.env.statetorch2oracle(all_states)
        rewards = self.proxy(all_oracle).detach().cpu().numpy()
        # TODO: state mask
        # state_mask = np.array(
        # [len(self.get_parents(s, False)[0]) > 0 or sum(s) == 0 for s in all_states]
        # )
        self._true_density = (
            rewards / rewards.sum(),
            rewards,
        )
        return self._true_density

    def get_all_terminating_states(self) -> List[List]:
        if hasattr(self.env, "get_all_terminating_states"):
            states = self.env.get_all_terminating_states()
            states = [state + [fid] for state in states for fid in range(self.n_fid)]
            return states
        else:
            return None

    def get_uniform_terminating_states(self, n_states: int) -> List[List]:
        if hasattr(self.env, "get_uniform_terminating_states"):
            return self.env.get_uniform_terminating_states(n_states)
        else:
            return None

    # def plot_reward_samples(self, samples, rewards = None, **kwargs):
    #     if hasattr(self.env, "plot_reward_samples"):
    #         return self.env.plot_reward_samples(samples, **kwargs)
    #     else:
    #         return None

    def plot_samples_frequency(self, samples, **kwargs):
        width = (self.n_fid) * 5
        fig, axs = plt.subplots(1, self.n_fid, figsize=(width, 5))
        for fid in range(0, self.n_fid):
            samples_fid = [sample[:-1] for sample in samples if sample[-1] == fid]
            if hasattr(self.env, "plot_samples_frequency"):
                axs[fid] = self.env.plot_samples_frequency(
                    samples_fid, axs[fid], "Fidelity {}".format(fid)
                )
            else:
                return None
        fig.suptitle("Frequency of Coordinates Sampled")
        plt.tight_layout()
        plt.show()
        return fig

    def get_cost(self, samples):
        fidelities = [sample[-1] for sample in samples]
        costs = [self.fidelity_costs[fid] for fid in fidelities]
        return costs
