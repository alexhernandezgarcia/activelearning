"""
Runnable script with hydra capabilities
"""
import sys
import random
import hydra
from omegaconf import OmegaConf

from gflownet.utils.common import flatten_config


@hydra.main(config_path="./config", config_name="main")
def main(config):
    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Log config
    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep="/")
    log_config = {"/".join(("config", key)): val for key, val in log_config.items()}

    ## Instantiate objects
    oracle = hydra.utils.instantiate(config.oracle)
    # TODO: Check if initialising env.proxy later breaks anything -- i.e., to check that nothin in the init() depends on the proxy
    env = hydra.utils.instantiate(config.env, oracle=oracle)
    # Note to Self: DataHandler needs env to make the train data. But DataHandler is required by regressor that's required by proxy that's required by env so we have to initalise env.proxy later on
    data_handler = hydra.utils.instantiate(config.dataset, env=env)
    # Regressor initialises a model which requires env-specific params so we pass the env-config with recursive=False os that the config as it is is passed instead of instantiating an object
    regressor = hydra.utils.instantiate(
        config.regressor,
        config_env=config.env,
        config_network=config.model,
        dataset=data_handler,
        _recursive_=False,
    )
    proxy = hydra.utils.instantiate(config.proxy, regressor=regressor)
    env.proxy = proxy
    # TODO: create logger and pass it to model
    gflownet = hydra.utils.instantiate(
        config.gflownet, env=env, buffer=config.env.buffer
    )

    for _ in range(config.al_n_rounds):
        regressor.fit()
        # TODO: rename gflownet to sampler once we have other sampling techniques ready
        gflownet.train()
        batch, times = gflownet.sample_batch(env, config.n_samples, train=False)
        queries = env.state2oracle(batch)
        energies = oracle(queries).tolist()
        data_handler.update_dataset(batch, energies)


if __name__ == "__main__":
    main()
    sys.exit()
