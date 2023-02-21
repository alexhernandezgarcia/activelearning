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
from regressor.gp import MultitaskGPRegressor

# ToyOracle,
# ,
#     plot_acquisition,
#     plot_context_points,
#     plot_predictions_oracle,
# )
from pathlib import Path
import pandas as pd


@hydra.main(config_path="./config", config_name="mf_amp_test")
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
            # required for toy oracle setups
            oracle=true_oracle,
            env=env,
            device=config.device,
            float_precision=config.float_precision,
        )
        oracles.append(oracle)
        if hasattr(oracle, "plot_true_rewards"):
            axs[fid - 1] = oracle.plot_true_rewards(
                env, axs[fid - 1], rescale=config.multifidelity.rescale
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

    # states = [[12, 5, 7], [3, 5, 12, 15]]
    # states_tensor = torch.tensor([[12, 5, 7, -2], [3, 5, 12, 15]])
    # tensor_policy_ohe = env.statetorch2policy(states_tensor)
    # numpy_policy_ohe = env.statebatch2policy(states)
    # print(tensor_policy_ohe)
    # print(numpy_policy_ohe)
    # torch_numpy_policy_ohe = torch.from_numpy(numpy_policy_ohe)
    # # check tensor_policy_ohe and numpy_policy_ohe for equality
    # assert torch.all(torch.eq(tensor_policy_ohe.detach().cpu(), torch_numpy_policy_ohe))

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
            config_model=config.model,
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
        print(f"\n Starting iteration {iter} of active learning")
        if logger:
            logger.set_context(iter)
        if N_FID == 1 or config.multifidelity.proxy == True:
            regressor.fit()
            if isinstance(regressor, MultitaskGPRegressor):
                fig = plot_gp_predictions(env, regressor, config.multifidelity.rescale)
                plt.tight_layout()
                plt.show()
                plt.close()
                logger.log_figure("gp_predictions", fig, use_context=True)
        proxy = hydra.utils.instantiate(
            config.proxy,
            regressor=regressor,
            device=config.device,
            float_precision=config.float_precision,
            logger=logger,
            oracle=oracles,
            env=env,
        )
        # fig = proxy.plot_context_points()
        # logger.log_figure("context_points", fig, True)
        # plot_context_points(env, proxy)
        # plot_acquisition(env, 0, proxy)
        # plot_acquisition(env, 1, proxy)
        # plot_acquisition(env, 2, proxy)
        # plot_predictions_oracle(env, 0)
        # plot_predictions_oracle(env, 1)
        # plot_predictions_oracle(env, 2)

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
            state_proxy = env.statebatch2proxy(states)
            if isinstance(state_proxy, list):
                state_proxy = torch.FloatTensor(state_proxy).to(config.device)
            scores = env.proxy(state_proxy)
            # to change desc/asc wrt higherIsBetter, and that should depend on proxy
            idx_pick = torch.argsort(scores, descending=True)[
                : config.n_samples
            ].tolist()
            picked_states = [states[i] for i in idx_pick]
            if N_FID > 1:
                picked_states, picked_fidelity = zip(
                    *[(state[:-1], state[-1]) for state in picked_states]
                )
                picked_fidelity = torch.LongTensor(picked_fidelity).to(config.device)
                picked_samples = env.env.statebatch2oracle(picked_states)
            else:
                picked_samples = env.statebatch2oracle(picked_states)

            if N_FID == 1:
                energies = env.oracle(picked_samples)
                picked_fidelity = None
            else:
                if config.env.proxy_state_format == "raw":
                    # Specifically for Branin
                    picked_states_tensor = torch.tensor(
                        picked_states, device=config.device, dtype=env.float
                    )
                    picked_states_tensor = (
                        picked_states_tensor / config.multifidelity.rescale
                    )
                    picked_states_oracle = picked_states_tensor.tolist()
                    energies = env.call_oracle_per_fidelity(
                        picked_states_oracle, picked_fidelity
                    )
                else:
                    energies = env.call_oracle_per_fidelity(
                        picked_samples, picked_fidelity
                    )

            if config.env.proxy_state_format == "ohe":
                gflownet.evaluate(
                    picked_samples, energies, data_handler.train_dataset["samples"]
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
                    list(picked_states), energies.tolist(), picked_fidelity
                )
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
