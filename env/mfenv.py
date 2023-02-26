from abc import abstractmethod
from typing import List, Tuple
import numpy.typing as npt
from torchtyping import TensorType
from gflownet.envs.base import GFlowNetEnv
import numpy as np
import torch
import matplotlib.pyplot as plt
import itertools
import copy


class MultiFidelityEnvWrapper(GFlowNetEnv):
    """
    Multi-fidelity environment for GFlowNet.
    Assumes same data transformation required for all oracles.
    Does not require the different oracles as scoring is performed by GFN not env
    """

    def __init__(
        self, env, n_fid, oracle, proxy_state_format=False, rescale=1, **kwargs
    ):
        # TODO: super init kwargs
        super().__init__(**kwargs)
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
            self.statebatch2proxy = self.statebatch2oracle_joined
            self.statetorch2proxy = self.statetorch2oracle_joined
        elif proxy_state_format == "ohe":
            self.statebatch2proxy = self.statebatch2policy
            self.statetorch2proxy = self.statetorch2policy
        elif proxy_state_format == "state":
            self.statebatch2proxy = self.statebatch2state
            self.statetorch2proxy = self.statebatch2state
            self.statetorch2oracle = self.state_from_statefid
            self.statebatch2oracle = self.state_from_statefid
        self.oracle = oracle
        self.fidelity_costs = self.set_fidelity_costs()
        self._test_traj_list = []
        self._test_traj_actions_list = []
        # rescale is used to rescale the state to the range of the oracle
        self.rescale = rescale
        self.reset()

    def set_proxy(self, proxy):
        self.proxy = proxy
        if hasattr(self, "proxy_factor"):
            return
        if self.proxy is not None and self.proxy.maximize is not None:
            # can be None for dropout regressor/UCB
            maximize = self.proxy.maximize
        elif self.oracle is not None:
            maximize = self.oracle[0].maximize
            print("\nAssumed that all Oracles have same maximize value.")
        else:
            raise ValueError("Proxy and Oracle cannot be None together.")
        if maximize:
            self.proxy_factor = 1.0
        else:
            self.proxy_factor = -1.0

    def statebatch2oracle_joined(self, states):
        # Specifically for Oracle MES and Oracle based MF Experiments
        states, fid_list = zip(*[(state[:-1], state[-1]) for state in states])
        state_oracle = self.env.statebatch2oracle(states)
        fidelities = torch.Tensor(fid_list).long().to(self.device).unsqueeze(-1)
        transformed_fidelities = torch.zeros_like(fidelities)
        for fid in range(self.n_fid):
            idx_fid = torch.where(fidelities == fid)[0]
            if idx_fid.shape[0] > 0:
                transformed_fidelities[idx_fid] = self.oracle[fid].fid
        states = torch.cat([state_oracle, transformed_fidelities], dim=-1)
        return states

    def statetorch2oracle_joined(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "oracle_dim"]:
        state_oracle = states[:, :-1]
        state_oracle = self.env.statetorch2oracle(state_oracle)
        fidelities = states[:, -1]
        transformed_fidelities = torch.zeros_like(fidelities)
        # assert torch.eq(
        #     torch.unique(fidelities), torch.arange(self.n_fid).to(fidelities.device)
        # ).all()
        for fid in range(self.n_fid):
            idx_fid = torch.where(fidelities == fid)[0]
            if idx_fid.shape[0] > 0:
                transformed_fidelities[idx_fid] = self.oracle[fid].fid
        states = torch.cat([state_oracle, transformed_fidelities.unsqueeze(-1)], dim=-1)
        return states

    def statebatch2state(self, states):
        if isinstance(states, TensorType) == False:
            states = torch.tensor(states, dtype=self.float, device=self.device)
        else:
            states = states.to(self.float).to(self.device)
        fidelities = states[:, -1]
        states = states[:, :-1]
        states = self.env.statebatch2state(states)
        # assert torch.eq(
        #     torch.unique(fidelities).sort()[0], torch.arange(self.n_fid).to(fidelities.device).sort()[0]
        # ).all()
        # TODO: Assertion breaks when all fidelities aren't sampled by the gflownet
        for fid in range(self.n_fid):
            idx_fid = torch.where(fidelities == fid)[0]
            fidelities[idx_fid] = self.oracle[fid].fid
        states = torch.cat([states, fidelities.unsqueeze(-1)], dim=-1)
        return states

    def state_from_statefid(self, states):
        if isinstance(states, TensorType) == False:
            states = torch.tensor(states, dtype=self.float, device=self.device)
        else:
            states = states.to(self.float).to(self.device)
        fidelities = states[:, -1]
        transformed_fidelities = torch.zeros_like(fidelities)
        states = states[:, :-1]
        states = self.env.statebatch2state(states)
        # assert torch.eq(
        #     torch.unique(fidelities).sort()[0],
        #     torch.arange(self.n_fid)
        #     .to(fidelities.device)
        #     .to(fidelities.dtype)
        #     .sort()[0],
        # ).all()
        for fid in range(self.n_fid):
            idx_fid = torch.where(fidelities == fid)[0]
            transformed_fidelities[idx_fid] = self.oracle[fid].fid
        return states, transformed_fidelities.unsqueeze(-1)

    def set_fidelity_costs(self):
        fidelity_costs = {}
        if hasattr(self.oracle[0], "fid") == False:
            print("\nOracle1 does not have fid. Re-assigning fid to all oracles. \n")
            # if anyone of the oracles dont have fid, fid is reassigned to all oracles
            for idx in range(self.n_fid):
                setattr(self.oracle[idx], "fid", idx)
        for idx in range(self.n_fid):
            fidelity_costs[self.oracle[idx].fid] = self.oracle[idx].cost
        return fidelity_costs

    def copy(self):
        env_copy = self.env.copy()
        class_var = self.__dict__.copy()
        class_var.pop("env", None)
        return MultiFidelityEnvWrapper(**class_var, env=env_copy)

    def call_oracle_per_fidelity(self, state_oracle, fidelities):
        if fidelities.ndim == 2:
            fidelities = fidelities.squeeze(-1)
        xx = torch.unique(fidelities).sort()[0]
        yy = torch.tensor(
            [self.oracle[fid].fid for fid in range(self.n_fid)],
            device=fidelities.device,
            dtype=fidelities.dtype,
        ).sort()[0]
        # TODO: Assertion breaks when all fidelities aren't sampled by gflownet
        # assert torch.eq(
        #     xx,
        #     yy).all()
        scores = torch.zeros(
            (fidelities.shape[0]), dtype=self.float, device=self.device
        )
        for fid in range(self.n_fid):
            idx_fid = torch.where(fidelities == self.oracle[fid].fid)[0]
            if isinstance(state_oracle, torch.Tensor):
                states = state_oracle[idx_fid]
                states = states.to(self.oracle[fid].device)
            else:
                chosen_state_index = torch.zeros(
                    scores.shape, dtype=self.float, device=self.device
                )
                chosen_state_index[idx_fid] = 1
                states = list(
                    itertools.compress(state_oracle, chosen_state_index.tolist())
                )
            if len(states) > 0:
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
            if isinstance(self.state, list):
                state = self.state.copy()
            elif isinstance(self.state, TensorType):
                state = self.state.clone()
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
        assert torch.eq(state.to(torch.long), state).all()
        state = state.to(torch.long)
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
        # each parent must be of the same length for self.tfloat to work
        # Can we have getparents return tensor instead of list?
        if self.env.do_state_padding and len(parents) > 0:
            max_parent_length = max([len(parent) for parent in parents])
            parents = [
                parent[:-1]
                + [self.env.invalid_state_element] * (max_parent_length - len(parent))
                + [parent[-1]]
                for parent in parents
            ]
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
            (states.shape[0], self.n_fid), dtype=self.float, device=self.device
        )
        # check if the last element of state is in the range of fidelity
        condition = torch.logical_and(
            states[:, -1] >= 0, states[:, -1] < self.n_fid
        ).unsqueeze(-1)
        index = torch.where(condition)[0]
        if index.size:
            fid_policy[index, states[index, -1].long()] = 1
        state_fid_policy = torch.cat((state_policy, fid_policy), dim=1)
        return state_fid_policy

    def statebatch2oracle(self, states: List[List]):
        states, fid_list = zip(*[(state[:-1], state[-1]) for state in states])
        state_oracle = self.env.statebatch2oracle(states)
        fidelities = torch.Tensor(fid_list).long().to(self.device).unsqueeze(-1)
        transformed_fidelities = torch.zeros_like(fidelities)
        for fid in range(self.n_fid):
            idx_fid = torch.where(fidelities == fid)[0]
            if idx_fid.shape[0] > 0:
                transformed_fidelities[idx_fid] = self.oracle[fid].fid
        # TODO: fix assertion -- runs into tuple error
        # assert torch.where(fid_torch == -1)[0].shape == 0
        return state_oracle, transformed_fidelities

    def statetorch2oracle(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "oracle_dim"]:
        state_oracle = states[:, :-1]
        state_oracle = self.env.statetorch2oracle(state_oracle)
        fidelities = states[:, -1]
        transformed_fidelities = torch.zeros_like(fidelities)
        # assert torch.eq(
        #     torch.unique(fidelities), torch.arange(self.n_fid).to(fidelities.device)
        # ).all()
        for fid in range(self.n_fid):
            idx_fid = torch.where(fidelities == fid)[0]
            if idx_fid.shape[0] > 0:
                transformed_fidelities[idx_fid] = self.oracle[fid].fid
        # state_fid = torch.cat((state_oracle, fid), dim=1)
        return state_oracle, transformed_fidelities

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
            if isinstance(states, TensorType):
                states = states.to(self.device).to(self.float)
            else:
                states = torch.tensor(states, dtype=self.float, device=self.device)
            fidelities = torch.zeros((len(states) * 3, 1)).to(states.device)
            for i in range(self.n_fid):
                fidelities[i * len(states) : (i + 1) * len(states), 0] = i
            states = states.repeat(self.n_fid, 1)
            state_fid = torch.cat([states, fidelities], dim=1)
            return state_fid.tolist()
        else:
            return None

    def get_uniform_terminating_states(self, n_states: int) -> List[List]:
        if hasattr(self.env, "get_uniform_terminating_states"):
            states = self.env.get_uniform_terminating_states(n_states)
            fidelities = torch.randint(0, self.n_fid, (n_states,)).tolist()
            state_fid = [state + [fid] for state, fid in zip(states, fidelities)]
            return state_fid
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
        plt.close()
        return fig

    def plot_reward_distribution(self, states, **kwargs):
        width = (self.n_fid) * 5
        fig, axs = plt.subplots(1, self.n_fid, figsize=(width, 5))
        for fid in range(0, self.n_fid):
            samples_fid = [state[:-1] for state in states if state[-1] == fid]
            if hasattr(self.env, "plot_reward_distribution"):
                axs[fid] = self.env.plot_reward_distribution(
                    states=samples_fid, ax=axs[fid], title="Fidelity {}".format(fid)
                )
            else:
                return None
        fig.suptitle("Rewards of Sequences Sampled")
        plt.tight_layout()
        plt.show()
        plt.close()
        return fig

    def get_cost(self, samples, fidelities=None):
        if fidelities is None:
            fidelities = [sample[-1] for sample in samples]
        fidelity_of__oracle = [self.oracle[int(fid)].fid for fid in fidelities]
        costs = [self.fidelity_costs[fid] for fid in fidelity_of__oracle]
        return costs

    def get_pairwise_distance(self, samples):
        if hasattr(self.env, "get_pairwise_distance"):
            return self.env.get_pairwise_distance(samples)
        else:
            return torch.zeros_like(samples)

    def get_distance_from_D0(self, samples, dataset_obs):
        if hasattr(self.env, "get_distance_from_D0"):
            return self.env.get_distance_from_D0(samples, dataset_obs)
        else:
            return torch.zeros_like(samples)

    def get_trajectories(
        self, traj_list, traj_actions_list, current_traj, current_actions
    ):
        # TODO: Optimize
        mf_traj_list = []
        mf_traj_actions_list = []
        traj_with_fidelity = current_traj[0]
        fidelity = traj_with_fidelity[-1]
        action_fidelity = (self.fid, fidelity)
        action = tuple(
            list(action_fidelity)
            + [0] * (self.action_max_length - len(action_fidelity))
        )
        current_traj = traj_with_fidelity[:-1]
        single_fid_traj_list, single_fid_traj_actions_list = self.env.get_trajectories(
            [], [], [current_traj], current_actions
        )
        for idx in range(len(single_fid_traj_actions_list)):
            for action_idx in range(len(single_fid_traj_actions_list[idx])):
                action = single_fid_traj_actions_list[idx][action_idx]
                action = tuple(
                    list(action) + [0] * (self.action_max_length - len(action))
                )
                single_fid_traj_actions_list[idx][action_idx] = action
        mf_traj_list = []
        mf_traj_actions_list = []
        for traj_idx in range(len(single_fid_traj_list)):
            num_traj_states = len(single_fid_traj_list[traj_idx])
            for idx in range(num_traj_states):
                trajs = copy.deepcopy(single_fid_traj_list[traj_idx])
                traj_actions = single_fid_traj_actions_list[traj_idx].copy()
                fidelity_traj = trajs[idx].copy()
                fidelity_traj.append(fidelity)
                trajs.insert(idx, fidelity_traj)
                traj_actions.insert(idx, action_fidelity)
                for j in range(idx):
                    trajs[j].append(fidelity)
                for k in range(idx + 1, len(trajs)):
                    trajs[k].append(-1)
                mf_traj_list.append(trajs)
                mf_traj_actions_list.append(traj_actions)
        # self._test_traj_list.append(mf_traj_list)
        # self._test_traj_actions_list.append(mf_traj_actions_list)
        return mf_traj_list, mf_traj_actions_list
