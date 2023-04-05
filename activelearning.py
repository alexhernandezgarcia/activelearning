"""
Runnable script with hydra capabilities
"""
import sys
import os
import random
from typing import List
import hydra
from omegaconf import OmegaConf
import torch
from env.mfenv import MultiFidelityEnvWrapper
from utils.multifidelity_toy import make_dataset
import matplotlib.pyplot as plt
from regressor.dkl import Tokenizer
import numpy as np
from utils.common import get_figure_plots


@hydra.main(config_path="./config", config_name="sf_aptamers")
def main(config):
    if config.logger.logdir.root == "./logs":
        cwd = os.getcwd()
        config.logger.logdir.root = cwd
    else:
        os.chdir(config.logger.logdir.root)

    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Set other random seeds
    set_seeds(config.seed)
    # Configure device count to avoid deserialise error
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # print(torch.cuda.device_count())

    print(
        "\n \tUser-Defined Warning: Oracles must be in increasing order of fidelity. \n \t Best oracle should be the last one in the config list."
    )

    # Logger
    logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)

    N_FID = len(config._oracle_dict)

    # check if key true_oracle is in config
    if "true_oracle" in config:
        true_oracle = hydra.utils.instantiate(
            config.true_oracle,
            device=config.device,
            float_precision=config.float_precision,
        )
    else:
        true_oracle = None

    env = hydra.utils.instantiate(
        config.env,
        oracle=true_oracle,
        device=config.device,
        float_precision=config.float_precision,
    )

    if hasattr(env, "vocab"):
        tokenizer = Tokenizer(env.vocab)
        env.set_tokenizer(tokenizer)
    else:
        tokenizer = None

    if hasattr(env, "rescale"):
        rescale = env.rescale
    else:
        rescale = 1.0
    oracles = []
    width = (N_FID) * 5
    fig, axs = plt.subplots(1, N_FID, figsize=(width, 5))
    do_figure = True
    for fid in range(1, N_FID + 1):
        oracle = hydra.utils.instantiate(
            config._oracle_dict[str(fid)],
            # required for toy grid oracle setups
            oracle=true_oracle,
            env=env,
            device=config.device,
            float_precision=config.float_precision,
        )
        oracles.append(oracle)
        if hasattr(oracle, "plot_true_rewards") and N_FID > 1:
            axs[fid - 1] = oracle.plot_true_rewards(env, axs[fid - 1], rescale=rescale)
        elif hasattr(oracle, "plot_true_rewards") and N_FID == 1:
            axs = oracle.plot_true_rewards(env, axs, rescale=rescale)
        else:
            do_figure = False
    if do_figure:
        plt.tight_layout()
        plt.show()
        plt.close()
        logger.log_figure("ground_truth_rewards", fig, use_context=False)

    if N_FID > 1:
        env = MultiFidelityEnvWrapper(
            env,
            n_fid=N_FID,
            oracle=oracles,
            proxy_state_format=config.env.proxy_state_format,
            rescale=rescale,
            device=config.device,
            float_precision=config.float_precision,
        )
        # Best fidelity
        env.env.oracle = oracles[-1]
    else:
        oracle = oracles[0]
        env.oracle = oracle

    if "model" in config:
        config_model = config.model
    else:
        config_model = None

    if hasattr(oracle, "modes"):
        modes = oracle.modes
    else:
        modes = None

    maximize = None

    if "budget" in config:
        BUDGET = config.budget
    else:
        if N_FID == 1:
            BUDGET = config.al_n_rounds * oracle.cost * config.n_samples
        else:
            BUDGET = oracles[-1].cost * config.al_n_rounds * config.n_samples

    if "proxy" in config and "mes" in config.proxy._target_.lower():
        is_mes = True
    else:
        is_mes = False

    if config.multifidelity.proxy == True:
        data_handler = hydra.utils.instantiate(
            config.dataset,
            env=env,
            logger=logger,
            oracle=oracle,
            device=config.device,
            float_precision=config.float_precision,
            rescale=rescale,
            is_mes=is_mes,
        )
    # check if path exists
    elif config.multifidelity.proxy == False:
        regressor = None
        if not os.path.exists(config.multifidelity.candidate_set_path):
            # makes context set for acquisition function
            make_dataset(
                env,
                oracles,
                N_FID,
                device=config.device,
                path=config.multifidelity.candidate_set_path,
            )

    cumulative_cost = 0.0
    cumulative_sampled_states = []
    cumulative_sampled_samples = []
    cumulative_sampled_energies = torch.tensor([], device=env.device, dtype=env.float)
    cumulative_sampled_fidelities = torch.tensor([], device=env.device, dtype=env.float)
    iter = 1

    env.reward_beta = env.reward_beta/env.beta_factor
    while cumulative_cost < BUDGET:
        env.reward_beta = env.reward_beta*env.beta_factor
        # for iter in range(1, config.al_n_rounds + 1):
        if config.multifidelity.proxy == True:
            # Moved in AL iter because of inducing point bug:
            # Different number of inducing points calculated by cholesky method in each iteration
            regressor = hydra.utils.instantiate(
                config.regressor,
                config_env=config.env,
                config_model=config_model,
                dataset=data_handler,
                device=config.device,
                maximize=oracle.maximize,
                float_precision=config.float_precision,
                _recursive_=False,
                logger=logger,
                tokenizer=tokenizer,
            )
        print(f"\nStarting iteration {iter} of active learning")
        if logger:
            logger.set_context(iter)
        if N_FID == 1 or config.multifidelity.proxy == True:
            regressor.fit()
            if hasattr(regressor, "evaluate_model"):
                metrics = {}
                (
                    fig,
                    test_rmse,
                    test_nll,
                    mode_rmse,
                    mode_nll,
                ) = regressor.evaluate_model(env, do_figure=config.do_figure)
                metrics.update(
                    {
                        "proxy_rmse_test": test_rmse,
                        "proxy_nll_test": test_nll,
                        "proxy_rmse_modes": mode_rmse,
                        "proxy_nll_modes": mode_nll,
                    }
                )
                if fig is not None:
                    plt.tight_layout()
                    plt.show()
                    plt.close()
                    if isinstance(fig, list) == False:
                        logger.log_figure("gp_predictions", fig, use_context=True)
                    else:
                        logger.log_figure("gp_predictions", fig[0], use_context=True)
                        logger.log_figure("uncertainties", fig[1], use_context=True)

                logger.log_metrics(metrics, use_context=False)
        if "proxy" in config:
            proxy = hydra.utils.instantiate(
                config.proxy,
                regressor=regressor,
                device=config.device,
                float_precision=config.float_precision,
                logger=logger,
                oracle=oracles,
                env=env,
            )
        else:
            proxy = None

        env.set_proxy(proxy)
        gflownet = hydra.utils.instantiate(
            config.gflownet,
            env=env,
            buffer=config.env.buffer,
            logger=logger,
            device=config.device,
            float_precision=config.float_precision,
        )
        # TODO: rename gflownet to sampler once we have other sampling techniques ready
        gflownet.train()
        if config.n_samples > 0 and config.n_samples <= 1e5:
            states, times = gflownet.sample_batch(
                env, config.n_samples * 5, train=False
            )
            if isinstance(states[0], list):
                states_tensor = torch.tensor(states)
            else:
                states_tensor = torch.vstack(states)
            states_tensor = states_tensor.unique(dim=0)
            if isinstance(states[0], list):
                # for all other envs, when we want list of lists
                states = states_tensor.tolist()
            else:
                # for AMP, when we want list of tensors
                states = list(states_tensor)
            state_proxy = env.statebatch2proxy(states)
            if isinstance(state_proxy, list):
                state_proxy = torch.FloatTensor(state_proxy).to(config.device)
            if proxy is not None:
                scores = env.proxy(state_proxy)
            else:
                scores, _ = regressor.get_predictions(env, states, denorm=True)
            num_pick = min(config.n_samples, len(states))
            if proxy is not None:
                maximize = proxy.maximize
            if maximize is None:
                if data_handler.target_factor == -1:
                    maximize = not oracle.maximize
                else:
                    maximize = oracle.maximize
            idx_pick = torch.argsort(scores, descending=maximize)[:num_pick].tolist()
            picked_states = [states[i] for i in idx_pick]

            if N_FID > 1:
                picked_samples, picked_fidelity = env.statebatch2oracle(picked_states)
                picked_energies = env.call_oracle_per_fidelity(
                    picked_samples, picked_fidelity
                )
            else:
                picked_samples = env.statebatch2oracle(picked_states)
                picked_energies = env.oracle(picked_samples)
                picked_fidelity = None

            if isinstance(picked_samples, torch.Tensor):
                # For when statebatch2oracle = statebatch2state in Grid and it returns a tensor of states instead
                picked_samples = picked_samples.tolist()

            cumulative_sampled_states.extend(picked_states)
            cumulative_sampled_samples.extend(picked_samples)
            cumulative_sampled_energies = torch.cat(
                (cumulative_sampled_energies, picked_energies)
            )
            if picked_fidelity is not None:
                cumulative_sampled_fidelities = torch.cat(
                    (cumulative_sampled_fidelities, picked_fidelity)
                )

            get_figure_plots(
                env,
                cumulative_sampled_states,
                cumulative_sampled_energies,
                cumulative_sampled_fidelities,
                logger,
                title="Cumulative Sampled Dataset",
                key="cum_sampled_dataset",
                use_context=True,
            )

            if hasattr(env, "get_cost"):
                cost_al_round = env.get_cost(picked_states, picked_fidelity)
                cumulative_cost += np.sum(cost_al_round)
                avg_cost = np.mean(cost_al_round)
                logger.log_metrics({"post_al_avg_cost": avg_cost}, use_context=False)
                # logger.log_metrics(
                #     {"post_al_cum_cost": cumulative_cost}, use_context=False
                # )
            else:
                cost_al_round = torch.ones(len(picked_states))
                if hasattr(oracle, "cost"):
                    cost_al_round = cost_al_round * oracle.cost
                avg_cost = torch.mean(cost_al_round).detach().cpu().numpy()
                cumulative_cost += torch.sum(cost_al_round).detach().cpu().numpy()
                logger.log_metrics({"post_al_avg_cost": avg_cost}, use_context=False)
                # logger.log_metrics(
                #     {"post_al_cum_cost": cumulative_cost}, use_context=False
                # )

            if config.env.proxy_state_format != "oracle":
                gflownet.evaluate(
                    cumulative_sampled_samples,
                    cumulative_sampled_energies,
                    oracle.maximize,
                    modes=modes,
                    dataset_states=data_handler.train_dataset["states"],
                    cumulative_cost=cumulative_cost,
                )
            if N_FID == 1 or config.multifidelity.proxy == True:
                data_handler.update_dataset(
                    picked_states, picked_energies.tolist(), picked_fidelity
                )

        del gflownet
        del proxy
        del regressor
        iter += 1


def set_seeds(seed):
    import torch
    import numpy as np

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
    sys.exit()
