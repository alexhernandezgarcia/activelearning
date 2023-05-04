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
from .grid import Grid
import pandas as pd


class MultiFidelityEnvWrapper(GFlowNetEnv):
    """
    Multi-fidelity environment for GFlowNet.
    Assumes same data transformation required for all oracles.
    Does not require the different oracles as scoring is performed by GFN not env
    """

    def __init__(
        self,
        env,
        n_fid,
        oracle,
        proxy_state_format=None,
        fid_embed="one_hot",
        fid_embed_dim=None,
        **kwargs
    ):
        # TODO: super init kwargs
        self.fid_embed = fid_embed
        if self.fid_embed != "one_hot" and fid_embed_dim is None:
            raise ValueError("fid_embed_dim cannot be None for non-OHE embedding")
        self.fid_embed_dim = fid_embed_dim
        super().__init__(**kwargs)
        self.env = env
        self.is_state_list = True
        if isinstance(self.env.source, TensorType):
            self.is_state_list = False
        self.n_fid = n_fid
        self.fid = self.env.eos + 1
        # Required for variables like reward_norm, reward_beta
        vars(self).update(vars(self.env))
        self.reset()
        self.action_space = self.get_actions_space()
        self.fixed_policy_output = self.get_fixed_policy_output()
        self.policy_input_dim = len(self.state2policy())
        self.policy_output_dim = len(self.fixed_policy_output)
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
            self.statetorch2proxy = self.statetorch2state
            if isinstance(self.env, Grid):
                # only for branin
                self.statetorch2oracle = self.state_from_statefid
                self.statebatch2oracle = self.state_from_statefid
        elif proxy_state_format == "state_fidIdx":
            self.statebatch2proxy = self.statebatch2state_longFid
            self.statetorch2proxy = self.statetorch2state_longFid
            if isinstance(self.env, Grid):
                # only for botorch synthetic functions like branin and hartmann
                self.statetorch2oracle = self.state_from_statefid
                self.statebatch2oracle = self.state_from_statefid
        else:
            raise ValueError("Invalid proxy_state_format")
        self.oracle = oracle
        self.fidelity_costs = self.set_fidelity_costs()
        if self.is_state_list:
            self.source = self.env.source + [-1]
        else:
            self.source = torch.cat((self.env.source, torch.tensor([-1])))
        self._test_traj_list = []
        self._test_traj_actions_list = []

    def write_samples_to_file(self, samples, path):
        samples = [self.state2readable(state) for state in samples]
        df = pd.DataFrame(samples, columns=["samples"])
        df.to_csv(path)

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

    def unpad_function(self, states_term):
        if hasattr(self.env, "unpad_function") == False:
            return states_term
        states_tensor = torch.tensor(states_term)
        states_tensor = states_tensor[:, :-1]
        state_XX = []
        for state in states_tensor:
            state = (
                state[: torch.where(state == self.padding_idx)[0][0]]
                if state[-1] == self.padding_idx
                else state
            )
            state_XX.append(state)
        return state_XX

    def statebatch2oracle_joined(self, states):
        # Specifically for Oracle MES and Oracle based MF Experiments
        states, fid_list = zip(*[(state[:-1], state[-1]) for state in states])
        state_oracle = self.env.statebatch2oracle(states)
        fidelities = torch.tensor(
            fid_list, dtype=torch.long, device=self.device
        ).unsqueeze(-1)
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

    def statebatch2state_longFid(self, states):
        state, fid = zip(*[[state[:-1], state[-1]] for state in states])
        states = self.env.statebatch2state(list(state))
        if self.is_state_list:
            fidelities = torch.tensor(
                fid, device=states.device, dtype=states.dtype
            ).unsqueeze(-1)
        else:
            fidelities = torch.vstack(fid).to(states.dtype).to(states.device)
        states = torch.cat([states, fidelities], dim=-1)
        return states

    def statetorch2state_longFid(self, states: TensorType["batch", "state_dim"]):
        state = states[:, :-1]
        fid = states[:, -1].unsqueeze(-1)
        state = self.env.statetorch2state(state)
        states = torch.cat([state, fid], dim=-1)
        return states

    def statebatch2state(self, states: List[TensorType["state_dim"]]):
        """
        Input: List of states where states can be tensors or lists
        """
        state, fid_list = zip(*[[state[:-1], state[-1]] for state in states])
        states = self.env.statebatch2state(list(state))
        if self.is_state_list:
            fidelities = torch.tensor(
                fid_list, device=states.device, dtype=states.dtype
            ).unsqueeze(-1)
        else:
            fidelities = torch.vstack(fid_list).to(states.dtype).to(states.device)
        # assert torch.eq(
        #     torch.unique(fidelities).sort()[0], torch.arange(self.n_fid).to(fidelities.device).sort()[0]
        # ).all()
        # TODO: Assertion breaks when all fidelities aren't sampled by the gflownet
        updated_fidelities = torch.zeros_like(fidelities)
        for fid in range(self.n_fid):
            idx_fid = torch.where(fidelities == fid)[0]
            if idx_fid.numel() > 0:
                updated_fidelities[idx_fid] = self.oracle[fid].fid
        states = torch.cat([states, updated_fidelities], dim=-1)
        return states

    def statetorch2state(self, states):
        if isinstance(states, TensorType) == False:
            raise ValueError("States should be a tensor")
            # states = torch.tensor(states, dtype=self.float, device=self.device)
        else:
            states = states.to(self.float).to(self.device)
        fidelities = states[:, -1]
        states = states[:, :-1]
        states = self.env.statetorch2state(states)
        fidelities = fidelities.to(states.dtype).to(states.device)
        # states = torch.cat([states, fidelities.unsqueeze(-1)], dim=-1)
        # assert torch.eq(
        #     torch.unique(fidelities).sort()[0], torch.arange(self.n_fid).to(fidelities.device).sort()[0]
        # ).all()
        # TODO: Assertion breaks when all fidelities aren't sampled by the gflownet
        updated_fidelities = torch.zeros_like(fidelities)
        for fid in range(self.n_fid):
            idx_fid = torch.where(fidelities == fid)[0]
            updated_fidelities[idx_fid] = self.oracle[fid].fid
        states = torch.cat([states, updated_fidelities.unsqueeze(-1)], dim=-1)
        return states

    def state_from_statefid(self, states):
        """
        Required specifically for Branin, as there the input to the oracle is the state itself.
        And state2oracle transformation cannot be applied because state2oracle brings the domain of points to [-1, 1] x [-1, 1] for the gird
        """
        if isinstance(states, TensorType) == False:
            states = torch.tensor(states, dtype=self.float, device=self.device)
        else:
            states = states.to(self.float).to(self.device)
        fidelities = states[:, -1]
        transformed_fidelities = torch.zeros_like(fidelities)
        states = states[:, :-1]
        states = self.env.statetorch2state(states)
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
            print(
                "\nUser-Defined Warning: Oracle1 does not have fid. Re-assigning fid to all oracles. \n"
            )
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
        if self.is_state_list:
            self.state = self.env.state + [fid]
        else:
            self.state = torch.cat([self.env.state, torch.tensor([fid])], dim=-1)
        # self.state = self.env.state + [fid]
        # Update the variables that were reset and are required in mf
        self.fid_done = self.state[-1] != -1
        self.done = self.env.done and self.fid_done
        self.id = self.env.id
        self.n_actions = 0
        return self

    def set_state(self, state, done):
        self.state = state
        self.env.state = state[:-1]
        self.done = done
        if done:
            self.fid_done = done
            self.env.done = done
        else:
            # Suppose done state was [2 3 4 0] with max_seq_length = 3
            # Then the two possible parents are: [2 3 4 -1] and [2 3 21 0]
            # In the first parent, env.done is True and fid_done is False
            # In the second parent, env.done is False and fid_done is True
            # For all subsequent parents of this parent, env.done will be False
            # For this particular parent, we have to check whether the fid was the last action
            # or something env-specific was the last action
            self.fid_done = self.state[-1] != -1
            if self.n_actions == self.get_max_traj_len() - 1:
                # If fid is incomplete then env is complete
                # If fid is complete then env is incomplete
                self.env.done = ~self.fid_done
            else:
                self.env.done = done
        return self

    def get_max_traj_len(self):
        return self.env.get_max_traj_len() + 1

    def get_mask_invalid_actions_forward(self, state=None, done=None, fid_done=None):
        if state is not None and fid_done is None:
            fid_done = state[-1] != -1
        elif state is None and fid_done is None:
            assert self.fid_done == (self.state[-1] != -1)
            fid_done = self.fid_done
        else:
            print("state, fid_done", state, fid_done)
            raise ValueError(
                "Either just state should be provided. Or both state and fid should be provided. Or nothing should be provided for consistency."
            )
        if state is None:
            if self.is_state_list:
                state = self.state.copy()
            else:
                state = self.state.clone()
                if isinstance(fid_done, torch.Tensor):
                    fid_done = fid_done.item()
        if done is None:
            done = self.done
        if done:
            try:
                assert self.env.done == True
            except:
                print(
                    "done is true but env.done is False: {}, {}".format(
                        done, self.env.done
                    )
                )
                raise Exception
            return [True for _ in range(len(self.action_space))]
        mask = self.env.get_mask_invalid_actions_forward(state[:-1])
        mask = mask + [fid_done for _ in range(self.n_fid)]
        return mask

    def state2policy(self, state: List = None):
        if state is None:
            if self.is_state_list:
                state = self.state.copy()
            else:
                state = self.state.clone()
        state_policy = self.env.state2policy(state[:-1])
        fid = state[-1]
        if self.fid_embed == "one_hot":
            fid_policy = np.zeros((self.n_fid), dtype=np.float32)
            if len(state) > 0 and fid != -1:
                fid_policy[fid] = 1
        elif self.fid_embed == "thermometer":
            per_fid_dimension = int(self.fid_embed_dim / self.n_fid)
            fid_policy = np.zeros((per_fid_dimension * self.n_fid), dtype=np.float32)
            if len(state) > 0 and fid != -1:
                fid_policy[0 : (fid + 1) * per_fid_dimension] = 1
        if self.is_state_list == False:
            state_policy = state_policy.squeeze(0).detach().cpu().numpy()
        state_fid_policy = np.concatenate((state_policy, fid_policy), axis=0)
        return state_fid_policy

    def statebatch2policy(self, states: List[List]) -> npt.NDArray[np.float32]:
        if not states:
            return np.array(states)
        state_policy, fid_list = zip(*[(state[:-1], state[-1]) for state in states])
        state_policy = self.env.statebatch2policy(state_policy)
        if self.fid_embed == "one_hot":
            fid_policy = np.zeros((len(states), self.n_fid), dtype=np.float32)
            fid_array = np.array(fid_list, dtype=np.int32)
            index = np.where(fid_array != -1)[0]
            if index.size:
                fid_policy[index, fid_array[index]] = 1
        elif self.fid_embed == "thermometer":
            per_fid_dimension = int(self.fid_embed_dim / self.n_fid)
            fid_policy = np.zeros(
                (len(states), per_fid_dimension * self.n_fid), dtype=np.float32
            )
            fid_array = np.array(fid_list, dtype=np.int32)
            fid_array = fid_array + 1
            upper_lim = fid_array * per_fid_dimension
            mask = np.arange(fid_policy.shape[1]) < upper_lim[:, np.newaxis]
            fid_policy[mask] = 1
        if self.is_state_list == False:
            state_policy = state_policy.squeeze(1).detach().cpu().numpy()
        state_fid_policy = np.concatenate((state_policy, fid_policy), axis=1)
        return state_fid_policy

    def policy2state(self, state_policy: List) -> List:
        policy_no_fid = state_policy[:, : -self.n_fid]
        state_no_fid = self.env.policy2state(policy_no_fid)
        if self.fid_embed == "one_hot":
            fid = np.argmax(state_policy[:, -self.n_fid :], axis=1)
        elif self.fid_embed == "thermometer":
            per_fid_dimension = int(self.fid_embed_dim / self.n_fid)
            fid = (
                np.argmax(state_policy[:, -self.n_fid * per_fid_dimension :], axis=1)
                / per_fid_dimension
            )
        state = state_no_fid + fid.to_list()
        return state

    def statetorch2policy(
        self, states: TensorType["batch", "state_dim"]
    ) -> TensorType["batch", "policy_input_dim"]:
        state_policy = states[:, :-1]
        state_policy = self.env.statetorch2policy(state_policy)
        if self.fid_embed == "one_hot":
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
        elif self.fid_embed == "thermometer":
            per_fid_dimension = int(self.fid_embed_dim / self.n_fid)
            fid_tensor = states[:, -1]
            fid_tensor = fid_tensor + 1
            upper_lim = fid_tensor * per_fid_dimension
            fid_policy = torch.zeros(
                (states.shape[0], per_fid_dimension * self.n_fid),
                dtype=self.float,
                device=self.device,
            )
            mask = torch.arange(
                fid_policy.shape[1], device=self.device, dtype=self.float
            ) < upper_lim.unsqueeze(-1)
            fid_policy[mask] = 1
        state_fid_policy = torch.cat((state_policy, fid_policy), dim=1)
        return state_fid_policy

    def state2readable(self, state=None):
        readable_state = self.env.state2readable(state[:-1])
        if isinstance(state[-1], torch.Tensor):
            fid = str(state[-1].detach().cpu().numpy())
        else:
            fid = str(state[-1])
        return readable_state + ";" + fid

    def statetorch2readable(self, state=None, **kwargs):
        try:
            assert torch.eq(state.to(torch.long), state).all()
        except:
            print(state)
            raise Exception("state is not integer")
        state = state.to(torch.long)
        readable_state = self.env.statetorch2readable(state[:-1], **kwargs)
        fid = str(state[-1].detach().cpu().numpy())
        # if isinstance(readable_state, list):
        #     # convert list to a string
        #     readable_state = str(readable_state)
        return readable_state + ";" + fid

    def readable2state(self, readable):
        fid = readable.split(";")[-1]
        readable_state = ";".join(readable.split(";")[:-1])
        state = self.env.readable2state(readable_state)
        if self.is_state_list:
            state = state + [int(fid)]
        else:
            state = torch.cat([state, torch.tensor([int(fid)])], dim=-1)
        return state

    def get_parents(self, state=None, done=None, action=None):
        if self.is_state_list:
            try:
                assert self.state[:-1] == self.env.state
            except:
                print(self.state[:-1], self.env.state)
                raise Exception("state and env.state is not equal in get_parents")
        else:
            try:
                assert torch.eq(self.state[:-1], self.env.state).all()
            except:
                print(self.state[:-1], self.env.state)
                raise Exception("state and env.state is not equal in get_parents")
        if state is None:
            if self.is_state_list:
                state = self.state.copy()
            else:
                state = self.state.clone()
        if done is None:
            done = self.done
        if (self.is_state_list and state == self.source and self.env.done == False) or (
            self.is_state_list == False
            and torch.eq(state, self.source).all()
            and self.env.done == False
        ):
            # The sf_env.done check has to be added because in the case of grid, sf_env.done can be True even at the source [0, 0]
            # But the parent_a would be eos
            return [], []
        parents_no_fid, actions = self.env.get_parents(state[:-1])
        if self.action_pad_length > 0:
            actions = [
                tuple(list(action) + [0] * (self.action_pad_length))
                for action in actions
            ]
        # If fidelity has not been chosen in the state, then fidelity has not been chosen in the parent as well
        if state[-1] == -1:
            if self.is_state_list:
                parents = [parent + [-1] for parent in parents_no_fid]
            else:
                parents_tensor = torch.vstack(parents_no_fid)
                fid_tensor = torch.ones((parents_tensor.shape[0], 1)) * -1
                parents = torch.cat([parents_tensor, fid_tensor], dim=-1)
                parents = list(parents)
        # If fidelity has been chosen, then possible parents incliude:
        # 1. The parent state with fidelity set to state's fidelity
        # 2. The parent with action as choosing that fidelity
        elif state[-1] != -1:
            if self.is_state_list:
                fid = state[-1]
                parents = [parent + [fid] for parent in parents_no_fid]
                fid_parent = state[:-1] + [-1]
            else:
                fid = state[-1].item()
                fid_tensor = torch.tensor([fid], device=state.device)
                parents = [
                    torch.cat([parent, fid_tensor], dim=-1) for parent in parents_no_fid
                ]
                fid_parent = torch.cat([state[:-1], torch.tensor([-1])], dim=-1)
            actions.append(tuple([self.fid, fid] + [0] * (self.action_max_length - 2)))
            parents.append(fid_parent)
        return parents, actions

    def step(self, action):
        if self.is_state_list:
            try:
                assert self.state[:-1] == self.env.state
            except:
                print(self.state[:-1], self.env.state)
                raise Exception("state and env.state is not equal in step")
        else:
            try:
                assert torch.eq(self.state[:-1], self.env.state).all()
            except:
                print(self.state[:-1], self.env.state)
                raise Exception("state and env.state is not equal in step")
        try:
            assert self.done == (self.env.done and self.fid_done)
        except:
            print(self.done, self.env.done, self.fid_done)
            raise Exception("done is not equal to (env.done and fid_done) in step")
        if self.state[-1] == -1:
            try:
                assert self.n_actions == self.env.n_actions
            except:
                print(self.n_actions, self.env.n_actions)
                raise Exception(
                    "n_actions is not equal to env.n_actions in step when fid has not been chosen"
                )
        else:
            try:
                assert self.n_actions == self.env.n_actions + 1
            except:
                print(self.n_actions, self.env.n_actions)
                raise Exception(
                    "n_actions is not equal to env.n_actions + 1 in step when fid has been chosen"
                )
        if self.done:
            parents, parents_actions = self.get_parents(
                state=self.state, done=self.done
            )
            for p in range(len(parents)):
                print(
                    parents[p],
                    parents_actions[p],
                    self.get_mask_invalid_actions_forward(state=parents[p], done=False),
                )
            print(
                self.state,
                action,
                self.get_mask_invalid_actions_forward(state=self.state, done=self.done),
            )
            raise ValueError("Action has been sampled despite environment being done")
            # return self.state, action, False
        if action[0] == self.fid:
            if self.fid_done == False:
                self.state[-1] = action[1]
                self.fid_done = True
                self.n_actions += 1
            else:
                parents, parents_actions = self.get_parents(
                    state=self.state, done=self.done
                )
                for p in range(len(parents)):
                    print(
                        parents[p],
                        parents_actions[p],
                        self.get_mask_invalid_actions_forward(
                            state=parents[p], done=False
                        ),
                    )
                print(
                    self.state,
                    action,
                    self.get_mask_invalid_actions_forward(
                        state=self.state, done=self.done
                    ),
                )
                raise ValueError("Fidelity has already been chosen.")
            assert self.fid_done == (self.state[-1] != -1)
            self.done = self.env.done and self.fid_done
            return self.state, action, True
        else:
            fid = self.state[-1]
            env_action = action[: -self.action_pad_length]
            state, action, valid = self.env.step(env_action)
            if valid:
                self.n_actions += 1
            if self.is_state_list:
                self.state = state + [fid]
            else:
                fid = torch.tensor([fid], device=state.device)
                self.state = torch.cat([state, fid], dim=-1)
            assert self.fid_done == (self.state[-1] != -1)
            if self.fid_done:
                assert self.n_actions == self.env.n_actions + 1
            else:
                assert self.n_actions == self.env.n_actions
            self.done = self.env.done and self.fid_done
            padded_action = tuple(list(action) + [0] * (self.action_pad_length))
            return self.state, padded_action, valid

    def statebatch2oracle(self, states: List[List]):
        states, fid_list = zip(*[(state[:-1], state[-1]) for state in states])
        state_oracle = self.env.statebatch2oracle(list(states))
        fidelities = torch.tensor(
            fid_list, dtype=torch.int64, device=self.device
        ).unsqueeze(-1)
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

    def plot_samples_frequency(self, samples, title=None, **kwargs):
        width = (self.n_fid) * 5
        fig, axs = plt.subplots(1, self.n_fid, figsize=(width, 5))
        for fid in range(0, self.n_fid):
            if isinstance(samples, TensorType):
                samples = samples.tolist()
                # samples_fid = samples[:, ]
            # else:
            samples_fid = [sample[:-1] for sample in samples if sample[-1] == fid]
            if hasattr(self.env, "plot_samples_frequency"):
                axs[fid] = self.env.plot_samples_frequency(
                    samples_fid, axs[fid], "Fidelity {}".format(fid)
                )
            else:
                return None
        # if title is None:
        # title = "Frequency of Coordinates Sampled"
        fig.suptitle(kwargs.get("title", "Frequency of Coordinates Sampled"))
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()
        plt.close()
        return fig

    def plot_reward_distribution(
        self, states=None, scores=None, fidelity=None, **kwargs
    ):
        width = (self.n_fid) * 5
        fig, axs = plt.subplots(1, self.n_fid, figsize=(width, 5))
        for fid in range(0, self.n_fid):
            if states is not None:
                samples_fid = [state[:-1] for state in states if state[-1] == fid]
                if hasattr(self.env, "plot_reward_distribution"):
                    axs[fid] = self.env.plot_reward_distribution(
                        states=samples_fid,
                        ax=axs[fid],
                        title="Fidelity {}".format(fid),
                        oracle=self.oracle[fid],
                    )
            elif scores is not None:
                title = kwargs.get("title", None)
                if "Initial Dataset" in title:
                    idx_fid = [i for i in range(len(scores)) if int(fidelity[i]) == fid]
                else:
                    idx_fid = [
                        i
                        for i in range(len(scores))
                        if fidelity[i] == self.oracle[fid].fid
                    ]
                if hasattr(self.env, "plot_reward_distribution") and len(idx_fid) != 0:
                    axs[fid] = self.env.plot_reward_distribution(
                        scores=scores[idx_fid],
                        ax=axs[fid],
                        title="Fidelity {}".format(fid),
                    )
            else:
                return None
        fig.suptitle(kwargs.get("title", "Reward Distribution"))
        plt.tight_layout()
        plt.show()
        plt.close()
        return fig

    def get_cost(self, samples, fidelities=None):
        if fidelities is None:
            fidelities = [self.oracle[int(sample[-1])].fid for sample in samples]
        if isinstance(fidelities, TensorType):
            if fidelities.ndim == 2:
                fidelities = fidelities.squeeze(-1)
            fidelities = fidelities.tolist()
        costs = [self.fidelity_costs[fid] for fid in fidelities]
        return costs

        # if fidelities is None:
        #     fidelities = [sample[-1] for sample in samples]
        #     fidelity_of_oracle = [self.oracle[int(fid)].fid for fid in fidelities]
        #     fidelities = fidelity_of_oracle
        # if isinstance(fidelities, TensorType):
        #     if fidelities.ndim == 2:
        #         fidelities = fidelities.squeeze(-1)
        #     fidelities = fidelities.tolist()
        # costs = [self.fidelity_costs[fid] for fid in fidelities]
        # return costs

    def get_pairwise_distance(self, samples_set1, samples_set2=None):
        if hasattr(self.env, "get_pairwise_distance"):
            return self.env.get_pairwise_distance(samples_set1, samples_set2)
        else:
            return torch.zeros(len(samples_set1))

    def get_distance_from_D0(self, samples, dataset_obs):
        if hasattr(self.env, "get_distance_from_D0"):
            dataset_obs = [sample[:-1] for sample in dataset_obs]
            return self.env.get_distance_from_D0(samples, dataset_obs)
        else:
            return torch.zeros(len(samples))

    def generate_fidelities(self, n_samples, config):
        if config.oracle_dataset.fid_type == "random":
            fidelities = torch.randint(low=0, high=self.n_fid, size=(n_samples, 1)).to(
                self.float
            )
        else:
            raise NotImplementedError
        return fidelities

    def initialize_dataset(self, config, n_samples, resume):
        train_scores = []
        test_scores = []
        train_samples = []
        test_samples = []

        if config.oracle_dataset is not None:
            # Train Path Given
            if config.oracle_dataset.train is not None:
                train = pd.read_csv(config.oracle_dataset.train.path)
                train_samples = train["samples"].values.tolist()
                if config.oracle_dataset.train.get_scores == False:
                    train_scores = train["energies"].values.tolist()
            # Test Path Given
            if config.oracle_dataset.test is not None:
                test = pd.read_csv(config.oracle_dataset.test.path)
                test_samples = test["samples"].values.tolist()
                if config.oracle_dataset.test.get_scores == False:
                    test_scores = test["energies"].values.tolist()

        # If neither were given, generate dataset
        if train_samples == [] and test_samples == []:
            # Make it generalised to generate states for any environment not just grid
            train_states = self.get_uniform_terminating_states(int(n_samples * 0.9))
            train_states = torch.tensor(train_states, dtype=self.float)
            test_states = self.get_uniform_terminating_states(int(n_samples * 0.1))
            test_states = torch.tensor(test_states, dtype=self.float)
            states = torch.cat((train_states, test_states), dim=0)
        else:
            samples = train_samples + test_samples
            if config.type == "sf":
                train_states = [
                    torch.tensor(self.env.readable2state(sample))
                    for sample in train_samples
                ]
                test_states = [
                    torch.tensor(self.env.readable2state(sample))
                    for sample in test_samples
                ]
                states = train_states + test_states

                # states = [
                # torch.tensor(self.env.readable2state(sample)) for sample in samples
                # ]
            else:
                train_states = [
                    torch.tensor(self.readable2state(sample))
                    for sample in train_samples
                ]
                test_states = [
                    torch.tensor(self.readable2state(sample)) for sample in test_samples
                ]
                states = train_states + test_states
                # train_states = torch.tensor(train_states, dtype=self.float)
                # test_states = torch.tensor(test_states, dtype=self.float)
                # states = [
                # torch.tensor(self.readable2state(sample)) for sample in samples
                # ]
            if train_states != []:
                train_states = torch.stack(train_states)
            if test_states != []:
                test_states = torch.stack(test_states)

            states = torch.stack(states)
            if config.type == "sf":
                fidelties = self.generate_fidelities(len(states), config)
                states = torch.cat([states, fidelties], dim=-1)

        scores = train_scores + test_scores
        # if isinstance(train_scores, List):
        #     scores = torch.tensor([train_scores + test_scores], dtype = self.float)
        # else:scores
        #      = torch.cat([train_scores, test_scores], dim=0)
        if scores == []:
            train_states_oracle_input = train_states.clone()
            train_state_oracle, fid = self.statetorch2oracle(train_states_oracle_input)
            train_scores = self.call_oracle_per_fidelity(train_state_oracle, fid)

            if test_states != []:
                test_states_oracle_input = test_states.clone()
                test_state_oracle, fid = self.statetorch2oracle(
                    test_states_oracle_input
                )
                test_scores = self.call_oracle_per_fidelity(test_state_oracle, fid)
            else:
                test_scores = torch.tensor([], dtype=self.float, device=self.device)
                test_states = torch.tensor([], dtype=self.float, device=self.device)

            scores = torch.cat([train_scores, test_scores], dim=0)

        if isinstance(scores, List):
            scores = torch.tensor(scores, dtype=self.float)
        if resume == False:
            return states, scores
        else:
            if isinstance(train_scores, List):
                train_scores = torch.tensor(train_scores, dtype=self.float)
            if isinstance(test_scores, List):
                test_scores = torch.tensor(test_scores, dtype=self.float)
            return train_states, train_scores, test_states, test_scores
