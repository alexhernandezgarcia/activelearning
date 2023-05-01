from gflownet.gflownet import GFlowNetAgent
import torch.nn as nn
from gflownet.utils.common import set_device, set_float_precision, torch2np
import numpy as np
import torch
import time
from torch.distributions import Categorical
from tqdm import tqdm
from gflownet.envs.base import Buffer
from collections import defaultdict

# from gflownet.gflownet import make_opt
from torchtyping import TensorType

# make self.forward_policy.model MLP and not self.forward_policy directly


class Policy:
    def __init__(self, config, env, device, float_precision):
        self.n_hid = config.n_hid
        self.n_layers = config.n_layers
        self.device = device
        self.float = float_precision
        self.model = self.make_mlp(
            [env.policy_input_dim]
            + [config.n_hid] * config.n_layers
            + [env.policy_output_dim + 1]
        )
        self.is_model = True
        if self.is_model:
            self.model.to(self.device)

    def make_mlp(self, l, act=nn.LeakyReLU(), tail=[]):
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
        ).to(dtype=self.float)

    def __call__(self, states):
        return self.model(states)


class PPOAgent(GFlowNetAgent):
    def __init__(
        self,
        env,
        seed,
        device,
        float_precision,
        optimizer,
        buffer,
        policy,
        ppo_num_epochs,
        ppo_epoch_size,
        logger,
        random_action_prob=0.0,
        ppo_clip=0.1,
        ppo_entropy_coef=1e-3,
        temperature_logits=1,
        active_learning=False,
        sample_only=False,
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
        self.ema_alpha = optimizer.ema_alpha
        self.early_stopping = optimizer.early_stopping
        # +1 for value function
        self.forward_policy = Policy(policy, self.env, self.device, self.float)
        if "checkpoint" in policy:
            self.logger.set_forward_policy_ckpt_path(policy.checkpoint)
            # TODO: re-write the logic and conditions to reload a model
            if False:
                self.forward_policy.load_state_dict(
                    torch.load(self.policy_forward_path)
                )
                print("Reloaded GFN forward policy model Checkpoint")
        else:
            self.logger.set_forward_policy_ckpt_path(None)
        self.ckpt_period = policy.ckpt_period
        if self.ckpt_period in [None, -1]:
            self.ckpt_period = np.inf
        self.opt, self.lr_scheduler = make_opt(
            self.forward_policy.model.parameters(), optimizer
        )
        self.use_context = active_learning
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        # Training
        self.temperature_logits = temperature_logits
        self.random_action_prob = random_action_prob
        self.clip_grad_norm = optimizer.clip_grad_norm
        # self.pct_offline = pct_offline
        # Metrics
        self.l1 = -1.0
        self.kl = -1.0
        self.jsd = -1.0
        self.buffer = Buffer(
            **buffer, env=self.env, make_train_test=not sample_only, logger=logger
        )
        # Train set statistics and reward normalization constant
        if self.buffer.train is not None:
            energies_stats_tr = [
                self.buffer.min_tr,
                self.buffer.max_tr,
                self.buffer.mean_tr,
                self.buffer.std_tr,
                self.buffer.max_norm_tr,
            ]
            self.env.set_energies_stats(energies_stats_tr)
            print("\nGFN Train data")
            print(f"\tMean score: {energies_stats_tr[2]}")
            print(f"\tStd score: {energies_stats_tr[3]}")
            print(f"\tMin score: {energies_stats_tr[0]}")
            print(f"\tMax score: {energies_stats_tr[1]}")
        else:
            energies_stats_tr = None
        if self.env.reward_norm_std_mult > 0 and energies_stats_tr is not None:
            self.env.reward_norm = self.env.reward_norm_std_mult * energies_stats_tr[3]
            self.env.set_reward_norm(self.env.reward_norm)
        # Test set statistics
        if self.buffer.test is not None:
            print("\nGFN Test Data")
            print(f"\tMean score: {self.buffer.mean_tt}")
            print(f"\tStd score: {self.buffer.std_tt}")
            print(f"\tMin score: {self.buffer.min_tt}")
            print(f"\tMax score: {self.buffer.max_tt}")

    def sample_actions(
        self,
        envs,
        sampling_method="policy",
        model=None,
        temperature=1.0,
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
        )  # actions is list of tuples
        return actions, logprobs

    def loss(self, it, batch, loginf=1000):
        loginf = self._tfloat([loginf])
        # (states,
        #     actions,
        #     logprobs,
        #     masks_s,
        #     advantages,
        #     rewards) = zip(*batch)
        idxs = torch.randint(low=0, high=len(batch), size=(self.batch_size,))
        # s, a, r, sp, d, lp, G, A = [torch.cat(i, 0) for i in zip(*[batch[i] for i in idxs])]

        # Concatenate lists of tensors in idx
        (
            states,
            actions,
            logprobs,
            states_sp,
            masks_sp,
            done,
            traj_id,
            state_id,
            masks_s,
            G,
            advantages,
        ) = [torch.cat(i, 0) for i in zip(*[batch[i] for i in idxs])]
        (
            _,
            _,
            _,
            _,
            _,
            is_terminal,
            _,
            _,
            _,
            rewards,
            _,
        ) = zip(*batch)
        is_terminal = torch.cat(is_terminal, 0)
        rewards = torch.cat(rewards, 0)[is_terminal.eq(1)]
        # zip(
        #     *batch
        # )
        # (states, actions, logprobs, masks_s, advantages, rewards, done, traj_id) = map(
        #     torch.cat,
        #     [states, actions, logprobs, masks_s, advantages, rewards, done, traj_id],
        # )
        # G = rewards.clone()
        # non_zero_indices = torch.nonzero(G).squeeze()
        # Replace zero elements with the immediately succeeding non-zero element
        # G[torch.cat([non_zero_indices[0].unsqueeze(0), non_zero_indices[:-1] + 1])] = G[
        #     non_zero_indices
        # ]

        policy_outputs = self.forward_policy(self.env.statetorch2policy(states))
        logits, values = policy_outputs[:, :-1], policy_outputs[:, -1]
        if masks_s is not None:
            logits[masks_s] = -loginf
        action_indices = (
            torch.tensor(
                [
                    self.env.action_space.index(tuple(action.tolist()))
                    for action in actions
                ]
            )
            .to(int)
            .to(self.device)
        )

        new_pol = Categorical(logits=logits)  # Categorical(logits: torch.Size([2, 3]))
        new_logprobs = new_pol.log_prob(action_indices)
        """
        was action a tensor of tuples?
        """
        ratio = torch.exp(new_logprobs - logprobs)
        surr1 = ratio * advantages
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * advantages
        )
        action_loss = -torch.min(surr1, surr2).mean()

        value_loss = 0.5 * (G - values).pow(2).mean()
        entropy = new_pol.entropy().mean()
        loss = action_loss + value_loss - entropy * self.entropy_coef

        # if not it % 100:
        #     print(G.mean())
        # if len(G[done.eq(1)])==0:
        #     # make a numpy array of nans
        #     rewards = np.nan
        # else:
        #     rewards = G[done.eq(1)]
        return (
            loss,
            action_loss,
            value_loss,
        ), rewards

        # policy_outputs = self.forward_policy(self.env.statetorch2policy(states[idxs]))
        # logits, values = policy_outputs[:, :-1], policy_outputs[:, -1]
        # if masks_s is not None:
        #     logits[masks_s[idxs]] = -loginf
        # action_indices = (
        #     torch.tensor(
        #         [
        #             self.action_space.index(tuple(action.tolist()))
        #             for action in actions[idxs]
        #         ]
        #     )
        #     .to(int)
        #     .to(self.device)
        # )

        # new_pol = Categorical(logits=logits)  # Categorical(logits: torch.Size([2, 3]))
        # new_logprobs = new_pol.log_prob(action_indices)
        # ratio = torch.exp(new_logprobs - logprobs)
        # surr1 = ratio * advantages[idxs]
        # surr2 = (
        #     torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        #     * advantages[idxs]
        # )
        # action_loss = -torch.min(surr1, surr2).mean()

        # value_loss = 0.5 * (G[idxs] - values).pow(2).mean()
        # entropy = new_pol.entropy().mean()
        # loss = action_loss + value_loss - entropy * self.entropy_coef

        # if not it % 100:
        #     print(G.mean())
        # return (loss, action_loss, value_loss,), G[
        #     done.eq(1)

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

        def _add_to_trajectory(
            trajs, curr_states, actions, logprobs, envs_sp, valids, train=True
        ):
            for s, action, logprob, env_sp, valid in zip(
                curr_states, actions, logprobs, envs_sp, valids
            ):
                if valid is False:
                    continue
                parents, _ = env.get_parents(action=action)
                mask_sp = env_sp.get_mask_invalid_actions_forward()
                if train:
                    trajs[env_sp.id].append(
                        # batch.append(
                        [
                            self._tfloat([s]),
                            self._tfloat([action]),
                            self._tfloat([logprob]),
                            self._tfloat([env_sp.state]),
                            self._tbool([mask_sp]),
                            self._tbool([env_sp.done]),
                            self._tlong([env_sp.id]),
                            self._tlong([env_sp.n_actions]),
                        ]
                    )

                else:
                    # TODO: to fix sampling
                    if env_sp.done:
                        trajs[env_sp.id].append(env_sp.state)
            return trajs

        times = {
            "actions_envs": 0.0,
        }
        trajs = defaultdict(list)
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
                        sampling_method="policy",
                        model=self.forward_policy,
                        temperature=1.0,
                    )
                else:
                    actions, logprobs = self.sample_actions(
                        envs,
                        sampling_method="policy",
                        model=self.forward_policy,
                        temperature=1.0,
                    )
            s = []
            # mask_s = []
            for env in envs:
                s.append(env.state)
                # mask_s.append(env.get_mask_invalid_actions_forward())
            # Update environments with sampled actions
            envs, actions, valids = self.step(envs, actions, is_forward=True)
            # Add to batch
            t0_a_envs = time.time()
            trajs = _add_to_trajectory(trajs, s, actions, logprobs, envs, valids, train)
            # Filter out finished trajectories
            envs = [env for env in envs if not env.done]
            t1_a_envs = time.time()
            times["actions_envs"] += t1_a_envs - t0_a_envs
            # if progress and n_samples is not None:
            #     print(f"{n_samples - len(envs)}/{n_samples} done")
        if train is False:
            batch = sum(trajs.values(), [])
            return batch, times

        for tau in trajs.values():
            (
                states_in_traj,
                actions_in_traj,
                logprobs_in_traj,
                states_sp_in_traj,
                masks_sp_in_traj,
                done_in_traj,
                traj_id,  # required for unpack_terminal_states
                state_id_in_traj,  # required for unpack_terminal_states
            ) = [torch.cat(i, 0) for i in zip(*tau)]
            rewards_in_traj = self.env.reward_torchbatch(
                states_sp_in_traj, done_in_traj
            )
            # state_id_in_traj = state_id_in_traj - state_id_in_traj.min() + 1
            masks_s = torch.cat([self.mask_source, masks_sp_in_traj[:-1]])

            # masks_s = torch.cat(
            # [
            #     masks_sp_in_traj[torch.where((state_id_in_traj == sid - 1))]
            #     if sid > 0
            #     else torch.zeros_like(masks_sp_in_traj[0])
            #     for sid in state_id_in_traj
            # ])
            # advantages = torch.zeros_like(rewards_in_traj)
            with torch.no_grad():  # masks not needed because dealing with values
                v_s = self.forward_policy(self.env.statetorch2policy(states_in_traj))[
                    :, -1
                ]
                v_sp = self.forward_policy(
                    self.env.statetorch2policy(states_sp_in_traj)
                )[:, -1]
            adv = rewards_in_traj + v_sp * torch.logical_not(done_in_traj) - v_s
            """rewards_in_traj should always have the last element as the reward of the terminal state"""
            for mask, i, A in zip(masks_s, tau, adv):
                i.append(mask.unsqueeze(0))  # mask_s
                i.append(
                    rewards_in_traj[-1].unsqueeze(0)
                )  # G, The return is always just the last reward, gamma is 1
                i.append(A.unsqueeze(0))  # A
        batch = sum(trajs.values(), [])
        """
        unpack batch
        states, actions, logprobs, states_sp, masks_sp, done, traj_id, state_id = [
            torch.cat(i, 0) for i in zip(*batch)
        ] #everything is a tensor -- might be an issue is state is a list instead
        # Compute rewards for all states (term and non-term)
        rewards = self.env.reward_torchbatch(states, done)
        # Shift state_id to [1, 2, ...]
        for tid in traj_id.unique():
            state_id[traj_id == tid] = (
                state_id[traj_id == tid] - state_id[traj_id == tid].min()
            ) + 1

        # Build parents forward masks from state masks
        masks_s = torch.cat(
            [
                masks_sp[torch.where((state_id == sid - 1) & (traj_id == pid))]
                if sid > 1
                else self.mask_source
                for sid, pid in zip(state_id, traj_id)
            ]
        )
        advantages = torch.zeros_like(rewards)
        for tid in traj_id.unique():
            traj_id_mask = traj_id == tid
            state_in_traj = states[traj_id_mask]
            state_sp_in_traj = states_sp[traj_id_mask]
            rewards_in_traj = rewards[traj_id_mask]
            with torch.no_grad(): #masks not needed because dealing with values
                v_s = self.forward_policy(self.env.statetorch2policy(state_in_traj))[
                    :, -1
                ]
                v_sp = self.forward_policy(
                    self.env.statetorch2policy(state_sp_in_traj)
                )[:, -1]
            adv = rewards_in_traj + v_sp * torch.logical_not(done[traj_id_mask]) - v_s
            advantages[traj_id_mask] = adv
            # for a in advantages:
                could not find a way to map the advantage to the corresponding batch element: mapping required because elements are added to batch ttsr number of times
        """
        # batch = [states, actions, logprobs, masks_s, advantages, rewards, done, traj_id]
        return batch, times

    def unpack_terminal_states(self, batch):
        """
        Unpacks the terminating states and trajectories of a batch and converts them
        to Python lists/tuples.
        """
        # TODO: make sure that unpacked states and trajs are sorted by traj_id (like
        # rewards will be)
        trajs = [[] for _ in range(self.batch_size)]
        states = [None] * self.batch_size
        for el in batch:
            traj_id = el[6][:1].item()
            state_id = el[7][:1].item()
            trajs[traj_id].append(tuple(el[1][0].tolist()))
            if bool(el[5].item()):
                states[traj_id] = tuple(el[0][0].tolist())
        trajs = [tuple(el) for el in trajs]
        return states, trajs

    def train(self):
        # Metrics
        all_losses = []
        all_visited = []
        loss_term_ema = None
        loss_flow_ema = None
        # Generate list of environments
        envs = [self.env.copy().reset() for _ in range(self.batch_size)]
        # Train loop
        pbar = tqdm(range(1, self.n_train_steps + 1), disable=not self.logger.progress)
        # PPOEdit: added for when first_iter = False for test
        x_sampled = None
        for it in pbar:
            # Test
            if self.logger.do_test(it):
                (
                    self.l1,
                    self.kl,
                    self.jsd,
                    self.corr,
                    x_sampled,
                    kde_pred,
                    kde_true,
                ) = self.test()
                self.logger.log_test_metrics(
                    self.l1, self.kl, self.jsd, self.corr, it, self.use_context
                )
            if self.logger.do_plot(it) and x_sampled is not None and len(x_sampled) > 0:
                figs = self.plot(x_sampled, kde_pred, kde_true)
                self.logger.log_plots(figs, it, self.use_context)
            t0_iter = time.time()
            data = []
            for j in range(self.sttr):
                batch, times = self.sample_batch(envs)
                data += batch
            for j in range(self.ttsr):
                losses, rewards = self.loss(
                    it * self.ttsr + j, data
                )  # returns (opt loss, *metrics)
                if not all([torch.isfinite(loss) for loss in losses]):
                    if self.logger.debug:
                        print("Loss is not finite - skipping iteration")
                    if len(all_losses) > 0:
                        all_losses.append([loss for loss in all_losses[-1]])
                else:
                    losses[0].backward()
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.parameters(), self.clip_grad_norm
                        )
                    if self.opt is not None:
                        # required for when fp is uniform
                        self.opt.step()
                        self.lr_scheduler.step()
                        self.opt.zero_grad()
                    all_losses.append([i.item() for i in losses])
            # Buffer
            # t0_buffer = time.time()
            states_term, trajs_term = self.unpack_terminal_states(batch)
            # check rewards is not NaN
            # if isinstance(rewards, TensorType)==False:
            #     proxy_vals = rewards
            # else:
            proxy_vals = self.env.reward2proxy(rewards).tolist()
            rewards = rewards.tolist()
            if self.logger.do_test(it) and hasattr(self.env, "get_cost"):
                costs = self.env.get_cost(states_term)
            else:
                costs = 0.0
            # self.buffer.add(states_term, trajs_term, rewards, proxy_vals, it)
            # self.buffer.add(
            # states_term, trajs_term, rewards, proxy_vals, it, buffer="replay"
            # )
            # t1_buffer = time.time()
            # times.update({"buffer": t1_buffer - t0_buffer})
            # Log
            if self.logger.lightweight:
                all_losses = all_losses[-100:]
                all_visited = states_term
            else:
                all_visited.extend(states_term)
            # Progress bar
            self.logger.progressbar_update(
                pbar, all_losses, rewards, self.jsd, it, self.use_context
            )
            # Train logs
            t0_log = time.time()
            if hasattr(self.env, "unpad_function"):
                unpad_function = self.env.unpad_function
            else:
                unpad_function = None
            self.logger.log_train(
                losses=losses,
                rewards=rewards,
                proxy_vals=proxy_vals,
                states_term=states_term,
                costs=costs,
                batch_size=len(data),
                logz=None,
                learning_rates=self.lr_scheduler.get_last_lr(),
                step=it,
                use_context=self.use_context,
                unpad_function=unpad_function,
            )
            t1_log = time.time()
            times.update({"log": t1_log - t0_log})
            # Save intermediate models
            t0_model = time.time()
            self.logger.save_models(self.forward_policy, None, step=it)
            t1_model = time.time()
            times.update({"save_interim_model": t1_model - t0_model})

            # Moving average of the loss for early stopping
            if loss_term_ema and loss_flow_ema:
                loss_term_ema = (
                    self.ema_alpha * losses[1] + (1.0 - self.ema_alpha) * loss_term_ema
                )
                loss_flow_ema = (
                    self.ema_alpha * losses[2] + (1.0 - self.ema_alpha) * loss_flow_ema
                )
                if (
                    loss_term_ema < self.early_stopping
                    and loss_flow_ema < self.early_stopping
                ):
                    break
            else:
                loss_term_ema = losses[1]
                loss_flow_ema = losses[2]
            # Log times
            t1_iter = time.time()
            times.update({"iter": t1_iter - t0_iter})
            self.logger.log_time(times, use_context=self.use_context)

        # Save final model
        self.logger.save_models(self.forward_policy, None, final=True)
        # Close logger
        if self.use_context == False:
            self.logger.end()


def make_opt(params, config):
    """
    Set up the optimizer
    """
    params = list(params)
    if not len(params):
        return None
    if config.method == "adam":
        opt = torch.optim.Adam(
            params,
            config.lr,
            betas=(config.adam_beta1, config.adam_beta2),
        )
    elif config.method == "msgd":
        opt = torch.optim.SGD(params, config.lr, momentum=config.momentum)
    # Learning rate scheduling
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        opt,
        step_size=config.lr_decay_period,
        gamma=config.lr_decay_gamma,
    )
    return opt, lr_scheduler
