"""
Runnable script with hydra capabilities
"""
import sys
import random
import hydra
from omegaconf import OmegaConf

# TODO: fix imports once gflownet package is ready
from src.gflownet.utils.common import flatten_config
from gflownet.src.gflownet.gflownet import GFlowNetAgent
from dataset import DataHandler


@hydra.main(config_path="./config", config_name="main")
def main(config):
    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Log config
    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep="/")
    log_config = {"/".join(("config", key)): val for key, val in log_config.items()}

    oracle = hydra.utils.instantiate(config.oracle)
    # TODO: Check if initialising env.proxy later breaks anything -- i.e., to check that nothin in the init() depends on the proxy
    env = hydra.utils.instantiate(config.env, oracle=oracle)
    # DataHandler needs env to make the train data
    # DataHandler needs oracle to score the created data
    # But DataHandler is required by regressor that's required by proxy that's required by env
    data_handler = hydra.utils.instantiate(config.dataset, oracle=oracle, env=env)
    # TODO: Initialise database_util and pass it to regressor
    # The regressor initialises a model which requires env-specific params so we pass the env-config
    regressor = hydra.utils.instantiate(
        config.model,
        config_network=config.network,
        config_env=config.env,
        dataset=data_handler,
    )
    # TODO: Create a proxy that takes the regressor.
    env.proxy = proxy
    # TODO: create logger and pass it to model
    gflownet = hydra.utils.instantiate(
        config.gflownet, env=env, buffer=config.env.buffer
    )
    # TODO: run the active learning training loop


if __name__ == "__main__":
    main()
    sys.exit()
