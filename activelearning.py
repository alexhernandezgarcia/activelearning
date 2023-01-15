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


@hydra.main(config_path="./config", config_name="main")
def main(config):
    cwd = os.getcwd()
    config.logger.logdir.root = cwd
    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Set other random seeds
    set_seeds(config.seed)
    # Log config
    # TODO: Move log config to Logger
    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep="/")
    log_config = {"/".join(("config", key)): val for key, val in log_config.items()}
    with open(cwd + "/config.yml", "w") as f:
        yaml.dump(log_config, f, default_flow_style=False)

    # Logger
    logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)

    ## Instantiate objects
    oracle = hydra.utils.instantiate(config.oracle)
    # TODO: Check if initialising env.proxy later breaks anything -- i.e., to check that nothin in the init() depends on the proxy
    env = hydra.utils.instantiate(config.env, oracle=oracle)
    # Note to Self: DataHandler needs env to make the train data. But DataHandler is required by regressor that's required by proxy that's required by env so we have to initalise env.proxy later on
    data_handler = hydra.utils.instantiate(config.dataset, env=env)
    # Regressor initialises a model which requires env-specific params so we pass the env-config with recursive=False os that the config as it is is passed instead of instantiating an object
    regressor = hydra.utils.instantiate(
        config.model,
        config_env=config.env,
        config_network=config.network,
        dataset=data_handler,
        _recursive_=False,
        logger=logger,
    )
    proxy = hydra.utils.instantiate(config.proxy, regressor=regressor)
    env.proxy = proxy
    # TODO: pass wandb logger to gflownet once it is compatible with wandb (i.e., once PR17 is merged)
    gflownet = hydra.utils.instantiate(
        config.gflownet,
        env=env,
        buffer=config.env.buffer,
        logger=logger,
        device=config.device,
    )

    for iter in range(1, config.al_n_rounds + 1):
        if logger:
            logger.set_context(iter)
        regressor.fit()
        # TODO: rename gflownet to sampler once we have other sampling techniques ready
        gflownet.train()
        batch, times = gflownet.sample_batch(env, config.n_samples, train=False)
        queries = env.state2oracle(batch)
        energies = oracle(queries).tolist()
        data_handler.update_dataset(batch, energies)


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
