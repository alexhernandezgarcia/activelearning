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


@hydra.main(config_path="./config", config_name="main")
def main(config):
    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Log config
    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep="/")
    log_config = {"/".join(("config", key)): val for key, val in log_config.items()}

    oracle = hydra.utils.instantiate(config.oracle)
    # TODO: Initialise database_util and pass it to regressor
    regressor = hydra.utils.instantiate(config.model, config_network= config.network)
    # TODO: Create a proxy that takes the regressor. Pass the proxy to env
    env = hydra.utils.instantiate(config.env, proxy=proxy)
    # TODO: create logger and pass it to model
    gflownet = hydra.utils.instantiate(config.gflownet, env=env, buffer=config.env.buffer)
    # TODO: run the active learning training loop

if __name__ == "__main__":
    main()
    sys.exit()