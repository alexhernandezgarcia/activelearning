from gflownet.gflownet import GFlowNetAgent
import torch.nn as nn
from gflownet.utils.common import set_device, set_float_precision, torch2np
import numpy as np
import torch
import time
from torch.distributions import Categorical


def make_mlp(l, act=nn.LeakyReLU(), tail=[]):
    """makes an MLP with no top layer activation"""
    return nn.Sequential(
        *(
            sum(
                [
                    [nn.Linear(i, o)] + ([act] if n < len(l) - 2 else [])
                    for n, (i, o) in enumerate(zip(l, l[1:]))
                ],
                [],
            )
            + tail
        )
    )


class PPOAgent(GFlowNetAgent):
    def __init__(
        self,
        env,
        seed,
        device,
        float_precision,
        optimizer,
        policy,
        ppo_num_epochs,
        ppo_epoch_size,
        logger,
        random_action_prob=0.0,
        ppo_clip=0.1,
        ppo_entropy_coef=1e-3,
        temperature_logits=1,
        active_learning=False,
        *args,
        **kwargs,
    ):
        # optimizer has clip_grad_norm=0
        # super().__init__(*args, **kwargs)
        # Seed
        self.rng = np.random.default_rng(seed)
        # Device
        self.device = set_device(device)
        # Float precision
        self.float = set_float_precision(float_precision)
        # Environment
        self.env = env
        self.mask_source = self._tbool([self.env.get_mask_invalid_actions_forward()])
        self.ppo_num_epochs = ppo_num_epochs
        self.clip_param = ppo_clip
        self.entropy_coef = ppo_entropy_coef
        self.ttsr = ppo_num_epochs
        self.sttr = ppo_epoch_size
        self.logger = logger
        self.n_train_steps = optimizer.n_train_steps
        self.batch_size = optimizer.batch_size
        # +1 for value function
        self.forward_policy = make_mlp(
            [self.env.policy_input_dim]
            + [policy.n_hid] * policy.n_layers
            + [self.env.policy_output_dim + 1]
        )
        self.use_context = active_learning
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        # Training
        self.temperature_logits = temperature_logits
        self.random_action_prob = random_action_prob
        # self.pct_offline = pct_offline
        # Metrics
        self.l1 = -1.0
        self.kl = -1.0
        self.jsd = -1.0

    def sample_actions(
        self,
        envs,
        times,
        sampling_method="policy",
        model=None,
        temperature=1.0,
        random_action_prob=0.0,
    ):
        """
        Samples one action on each environment of a list.

        Args
        ----
        envs : list of GFlowNetEnv or derived
            A list of instances of the environment

        times : dict
            Dictionary to store times

        sampling_method : string
            - model: uses current forward to obtain the sampling probabilities.
            - uniform: samples uniformly from the action space.

        model : torch model
            Model to use as policy if sampling_method="policy"

        is_forward : bool
            True if sampling is forward. False if backward.

        temperature : float
            Temperature to adjust the logits by logits /= temperature
        """
        # TODO: implement backward sampling from forward policy as in old
        # backward_sample.
        if sampling_method == "random":
            random_action_prob = 1.0
        if not isinstance(envs, list):
            envs = [envs]
        # Build states and masks
        states = [env.state for env in envs]
        mask_invalid_actions = self._tbool(
            [env.get_mask_invalid_actions_forward() for env in envs]
        )
        # Build policy outputs

        policy_outputs = model(self._tfloat(self.env.statebatch2policy(states)))
        # Skip v from policy outputs
        policy_outputs = policy_outputs[:, :-1]
        # Sample actions from policy outputs
        actions, logprobs = self.env.sample_actions(
            policy_outputs,
            sampling_method,
            mask_invalid_actions,
            temperature,
        )
        return actions, logprobs

    def loss(self, it, batch, loginf=1000):
        loginf = self._tfloat([loginf])
        # Unpack batch
        (
            states,
            actions,
            logprobs,
            sp,
            done,
            traj_id,
            state_id,
            masks_s_old,
            masks_sp,
        ) = zip(*batch)

        idxs = torch.randint(0, len(batch), self.mbsize)
        # Concatenate lists of tensors in idx
        (
            states,
            actions,
            logprobs,
            sp,
            done,
            traj_id,
            state_id,
            masks_s_old,
            masks_sp,
        ) = map(
            torch.cat,
            [
                states,
                actions,
                logprobs,
                sp,
                done,
                traj_id,
                state_id,
                masks_s_old,
                masks_sp,
            ],
        )
        # Shift state_id to [1, 2, ...]
        for tid in traj_id.unique():
            state_id[traj_id == tid] = (
                state_id[traj_id == tid] - state_id[traj_id == tid].min()
            ) + 1
        # Compute rewards
        rewards = self.env.reward_torchbatch(
            states, done
        )  # should be for all states (term and non-term)

        # Build parents forward masks from state masks
        masks_s = torch.cat(
            [
                masks_sp[torch.where((state_id == sid - 1) & (traj_id == pid))]
                if sid > 1
                else self.mask_source
                for sid, pid in zip(state_id, traj_id)
            ]
        )

        # s, a, r, sp, d, lp, G, A = [torch.cat(i, 0) for i in zip(*[batch[i] for i in idxs])]
        policy_outputs = self.forward_policy(self.env.statetorch2policy(states[idxs]))
        logits, values = policy_outputs[:, :-1], policy_outputs[:, -1]
        actions, new_logprobs = self.env.sample_actions(
            logits,
            mask_invalid_actions=masks_s,
            temperature=1.0,
        )

        ratio = torch.exp(new_logprobs - logprobs)

        surr1 = ratio * A
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * A
        action_loss = -torch.min(surr1, surr2).mean()
        G = torch.zeros_like(rewards)
        non_zero_indices = torch.nonzero(rewards).squeeze()
        # Replace zero elements with the immediately succeeding non-zero element
        G[torch.cat([non_zero_indices[0].unsqueeze(0), non_zero_indices[:-1] + 1])] = G[
            non_zero_indices
        ]
        # G should be terminal rewards of the trajectory
        # state -> traj_id and use that traj_id to get reward[traj_id] which will be G
        value_loss = 0.5 * (G - values).pow(2).mean()
        entropy = new_pol.entropy().mean()
        if not it % 100:
            print(G.mean())
        return (
            action_loss + value_loss - entropy * self.entropy_coef,
            action_loss,
            value_loss,
        ), rewards

    def sample_batch(
        self, envs, n_samples=None, train=True, model=None, progress=False
    ):
        """
        Builds a batch of data

        if train == True:
            Each item in the batch is a list of 7 elements (all tensors):
                - [0] s, the state
                - [1] a, the action
                - [2] lp, log_prob of action
                - [3] mask_s
                - [4] sp, the next state
                - [5] mask_sp
                - [6] done [True, False]
                - [7] traj id: identifies each trajectory
                - [8] state id: identifies each state within a traj
        else:
            Each item in the batch is a list of 1 element:
                - [0] the states (state)

        Args
        ----
        """

        def _add_to_batch(
            batch, curr_states, actions, logprobs, masks_s, envs_sp, valids, train=True
        ):
            for s, action, logprob, mask_s, env_sp, valid in zip(
                curr_states, actions, logprobs, masks_s, envs_sp, valids
            ):
                if valid is False:
                    continue
                parents, _ = env.get_parents(action=action)
                mask_sp = env_sp.get_mask_invalid_actions_forward()
                if train:
                    batch.append(
                        [
                            self._tfloat([s]),
                            self._tfloat([action]),
                            self._tfloat([logprob]),
                            self._tbool([mask_s]),
                            self._tfloat([env_sp.state]),
                            self._tbool([mask_sp]),
                            self._tbool([env_sp.done]),
                            self._tlong([env_sp.id]),
                            self._tlong([env_sp.n_actions]),
                        ]
                    )

                else:
                    if env.done:
                        batch.append(env.state)
            return batch

        times = {
            "actions_envs": 0.0,
        }
        batch = []
        if isinstance(envs, list):
            envs = [env.reset(idx) for idx, env in enumerate(envs)]
        elif n_samples is not None and n_samples > 0:
            envs = [self.env.copy().reset(idx) for idx in range(n_samples)]
        else:
            return None, None

        while envs:
            # Sample forward actions
            with torch.no_grad():
                if train is False:
                    actions, logprobs = self.sample_actions(
                        envs,
                        times,
                        sampling_method="policy",
                        model=self.forward_policy,
                        is_forward=True,
                        temperature=1.0,
                        random_action_prob=self.random_action_prob,
                    )
                else:
                    actions, logprobs = self.sample_actions(
                        envs,
                        times,
                        sampling_method="policy",
                        model=self.forward_policy,
                        is_forward=True,
                        temperature=self.temperature_logits,
                        random_action_prob=self.random_action_prob,
                    )
            s = []
            mask_s = []
            for env in envs:
                s.append(env.state)
                mask_s.append(env.get_mask_invalid_actions_forward())
            # Update environments with sampled actions
            envs_sp, actions, valids = self.step(envs, actions, is_forward=True)
            # Add to batch
            t0_a_envs = time.time()
            batch = _add_to_batch(
                batch, s, actions, logprobs, mask_s, envs_sp, valids, train
            )
            # Filter out finished trajectories
            envs = [env for env in envs_sp if not env.done]
            t1_a_envs = time.time()
            times["actions_envs"] += t1_a_envs - t0_a_envs
            if progress and n_samples is not None:
                print(f"{n_samples - len(envs)}/{n_samples} done")
        return batch, times
