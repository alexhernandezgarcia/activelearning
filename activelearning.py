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
import numpy as np
import torch
from env.mfenv import MultiFidelityEnvWrapper
from utils.multifidelity_toy import ToyOracle, make_dataset
import operator


@hydra.main(config_path="./config", config_name="mf_debug_test")
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

    if config.multifidelity.toy:
        N_FID = config.multifidelity.n_fid
    else:
        N_FID = len(config._oracle_dict)

    oracles = []
    if config.multifidelity.toy:
        oracle = hydra.utils.instantiate(
            config.oracle, device=config.device, float_precision=config.float_precision
        )
        # target fidelity must be the last fidelity
        # config._noise_dict = dict( sorted(config._noise_dict.items(), key=operator.itemgetter(1),reverse=True))
        for fid in range(1, N_FID + 1):
            toy = ToyOracle(
                oracle,
                config._noise_dict[str(fid)],
                config.device,
                config.float_precision,
            )
            oracles.append(toy)
    else:
        for fid in range(1, N_FID + 1):
            oracle = hydra.utils.instantiate(
                config._oracle_dict[str(fid)],
                device=config.device,
                float_precision=config.float_precision,
            )
            oracles.append(oracle)

    env = hydra.utils.instantiate(
        config.env,
        oracle=oracle,
        device=config.device,
        float_precision=config.float_precision,
    )

    if N_FID > 1:
        env = MultiFidelityEnvWrapper(
            env, n_fid=N_FID, oracle=oracles, is_oracle_proxy=config.multifidelity.toy
        )

    if config.multifidelity.toy == False:
        data_handler = hydra.utils.instantiate(
            config.dataset, env=env, logger=logger, oracle=oracle
        )
        regressor = hydra.utils.instantiate(
            config.regressor,
            config_env=config.env,
            config_model=config.model,
            dataset=data_handler,
            device=config.device,
            _recursive_=False,
            logger=logger,
        )
    # check if path exists
    elif config.multifidelity.toy and not os.path.exists(
        config.multifidelity.candidate_set_path
    ):
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
        if config.multifidelity.toy == False:
            regressor.fit()
            proxy = hydra.utils.instantiate(
                config.proxy,
                regressor=regressor,
                device=config.device,
                float_precision=config.float_precision,
            )
        else:
            # proxy is used to get rewards and in oracle steup, we get rewards by calling separate oracle for each state
            proxy = hydra.utils.instantiate(
                config.proxy,
                device=config.device,
                float_precision=config.float_precision,
                n_fid=N_FID,
                logger=logger,
                oracle=oracles,
                env=env,
            )
        env.proxy = proxy
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
            scores = env.proxy(env.statebatch2proxy(states))
            idx_pick = torch.argsort(scores, descending=True)[
                : config.n_samples
            ].tolist()
            picked_states = [states[i] for i in idx_pick]
            picked_samples = env.statebatch2oracle(picked_states)

            energies = env.call_oracle_per_fidelity(picked_samples)
            if config.multifidelity.toy == False:
                gflownet.evaluate(
                    picked_samples, energies, data_handler.train_dataset["samples"]
                )
                # dataset will eventually store in proxy-format so states are sent to avoid the readable2state conversion
                # batch, fid, times = gflownet.sample_batch(env, config.n_samples, train=False)
                # queries = [env.state2oracle[f](b) for f, b in zip(fid, batch)]
                # energies = [oracles[f](q) for f, q in zip(fid, queries)]
                data_handler.update_dataset(
                    picked_states, energies.detach().cpu().numpy()
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
