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
    N_FID = len(config._oracle_dict)
    oracles = []
    for fid in range(1, N_FID+1):
        oracle = hydra.utils.instantiate(config._oracle_dict[str(fid)])
        oracles.append(oracle)

    # Assume oracle list has highest fidelity first so that is used to calculate the true density
    env = hydra.utils.instantiate(config.env, oracle=oracles[0])
    data_handler = hydra.utils.instantiate(config.dataset, env=env)
    regressor = hydra.utils.instantiate(
        config.model,
        config_env=config.env,
        config_network=config.network,
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
        batch, fid, times = gflownet.sample_batch(env, config.n_samples, train=False)
        queries = [env.state2oracle[f](b) for f, b in zip(fid, batch)]
        energies = [oracles[f](q) for f, q in zip(fid, queries)]
        data_handler.update_dataset(batch, energies)


if __name__ == "__main__":
    main()
    sys.exit()
