"""
Runnable script with hydra capabilities
"""
import sys
import os
import random
import hydra
from omegaconf import OmegaConf
import yaml
from gflownet.utils.common import flatten_config
import torch
from env.mfenv import MultiFidelityEnvWrapper
from utils.multifidelity_toy import make_dataset, plot_gp_predictions
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np


@hydra.main(config_path="./config", config_name="mf_corners_oracle_only")
def main(config):
    cwd = os.getcwd()
    config.logger.logdir.root = cwd
    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Set other random seeds
    set_seeds(config.seed)
    # Configure device count to avoid deserialise error
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # print(torch.cuda.device_count())
    # Log config
    # TODO: Move log config to Logger
    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep="/")
    log_config = {"/".join(("config", key)): val for key, val in log_config.items()}
    with open(cwd + "/config.yml", "w") as f:
        yaml.dump(log_config, f, default_flow_style=False)

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
            axs[fid - 1] = oracle.plot_true_rewards(
                env, axs[fid - 1], rescale=config.multifidelity.rescale
            )
        elif hasattr(oracle, "plot_true_rewards") and N_FID == 1:
            axs = oracle.plot_true_rewards(
                env, axs, rescale=config.multifidelity.rescale
            )
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
            rescale=config.multifidelity.rescale,
        )
    else:
        oracle = oracles[0]
        env.oracle = oracle

    if "model" in config:
        config_model = config.model
    else:
        config_model = None

    if N_FID == 1 or config.multifidelity.proxy == True:
        data_handler = hydra.utils.instantiate(
            config.dataset,
            env=env,
            logger=logger,
            oracle=oracle,
            device=config.device,
            float_precision=config.float_precision,
            rescale=config.multifidelity.rescale,
        )
        regressor = hydra.utils.instantiate(
            config.regressor,
            config_env=config.env,
            config_model=config_model,
            dataset=data_handler,
            device=config.device,
            float_precision=config.float_precision,
            _recursive_=False,
            logger=logger,
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

    for iter in range(1, config.al_n_rounds + 1):
        print(f"\nStarting iteration {iter} of active learning")
        if logger:
            logger.set_context(iter)
        if N_FID == 1 or config.multifidelity.proxy == True:
            regressor.fit()
            if hasattr(regressor, "evaluate_model"):
                metrics = {}
                fig, rmse, nll = regressor.evaluate_model(
                    env, config.multifidelity.rescale
                )
                metrics.update(
                    {
                        "proxy_rmse": rmse,
                        "proxy_nll": nll,
                    }
                )
                if fig is not None:
                    plt.tight_layout()
                    plt.show()
                    plt.close()
                    logger.log_figure("gp_predictions", fig, use_context=True)
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
            if proxy is not None:
                if env.do_state_padding == False:
                    # TODO: only possible if all states are same length
                    states_tensor = torch.tensor(states)
                    states_tensor = states_tensor.unique(dim=0)
                    states = states_tensor.tolist()
                state_proxy = env.statebatch2proxy(states)
                if isinstance(state_proxy, list):
                    state_proxy = torch.FloatTensor(state_proxy).to(config.device)
                scores = env.proxy(state_proxy)
                # to change desc/asc wrt higherIsBetter, and that should depend on proxy
                num_pick = min(config.n_samples, len(states))
                idx_pick = torch.argsort(scores, descending=True)[:num_pick].tolist()
                picked_states = [states[i] for i in idx_pick]
            else:
                print(
                    "\nSince there is no proxy, we simply take the first {n_samples} number of states sampled (instead of sorting by energy)."
                )
                picked_states = states[: config.n_samples]

            if N_FID > 1:
                picked_samples, picked_fidelity = env.statebatch2oracle(picked_states)
                energies = env.call_oracle_per_fidelity(picked_samples, picked_fidelity)
            else:
                picked_samples = env.statebatch2oracle(picked_states)
                energies = env.oracle(picked_samples)
                picked_fidelity = None

            if config.env.proxy_state_format != "oracle":
                gflownet.evaluate(
                    picked_samples, energies, data_handler.train_dataset["states"]
                )

            df = pd.DataFrame(
                {
                    "readable": [env.state2readable(s) for s in picked_states],
                    "energies": energies.cpu().numpy(),
                }
            )
            df = df.sort_values(by=["energies"])
            path = logger.logdir / Path("gfn_samples.csv")
            df.to_csv(path)
            if N_FID == 1 or config.multifidelity.proxy == True:
                data_handler.update_dataset(
                    picked_states, energies.tolist(), picked_fidelity
                )
            if hasattr(env, "get_cost"):
                avg_cost = np.mean(env.get_cost(picked_states, picked_fidelity))
                logger.log_metrics({"post_al_cost": avg_cost}, use_context=False)
        del gflownet
        del proxy


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
