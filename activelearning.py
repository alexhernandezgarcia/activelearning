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
# from memory_profiler import profile

@hydra.main(config_path="./config", config_name="main")
# @profile
def main(config):
    cwd = os.getcwd()
    config.logger.logdir.root = cwd
    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Set other random seeds
    set_seeds(config.seed)
    # Configure device count to avoid derserialise error
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print(torch.cuda.device_count())
    # Log config
    # TODO: Move log config to Logger
    log_config = flatten_config(OmegaConf.to_container(config, resolve=True), sep="/")
    log_config = {"/".join(("config", key)): val for key, val in log_config.items()}
    with open(cwd + "/config.yml", "w") as f:
        yaml.dump(log_config, f, default_flow_style=False)

    # Logger
    logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)

    ## Instantiate objects
    oracle = hydra.utils.instantiate(config.oracle, device=config.device)
    # TODO: Check if initialising env.proxy later breaks anything -- i.e., to check that nothin in the init() depends on the proxy
    env = hydra.utils.instantiate(config.env, oracle=oracle)
    # Note to Self: DataHandler needs env to make the train data. But DataHandler is required by regressor that's required by proxy that's required by env so we have to initalise env.proxy later on
    data_handler = hydra.utils.instantiate(
        config.dataset, env=env, logger=logger, oracle=oracle
    )
    # Regressor initialises a model which requires env-specific params so we pass the env-config with recursive=False os that the config as it is is passed instead of instantiating an object
    regressor = hydra.utils.instantiate(
        config.regressor,
        config_env=config.env,
        config_model=config.model,
        dataset=data_handler,
        _recursive_=False,
        logger=logger,
    )

    for iter in range(1, config.al_n_rounds + 1):
        print(f"\n Starting iteration {iter} of active learning")
        if logger:
            logger.set_context(iter)
        regressor.fit()
        proxy = hydra.utils.instantiate(config.proxy, regressor=regressor)
        env.proxy = proxy
        gflownet = hydra.utils.instantiate(
            config.gflownet,
            env=env,
            buffer=config.env.buffer,
            logger=logger,
            device=config.device,
        )
        # TODO: rename gflownet to sampler once we have other sampling techniques ready
        gflownet.train()
        if config.n_samples > 0 and config.n_samples <= 1e5:
            states, times = gflownet.sample_batch(
                env, config.n_samples * 5, train=False
            )
            # TODO: Change state2oracle after continous branch merge
            scores = env.proxy(env.state2proxy(states))
            idx_pick = np.argsort(scores)[::-1][: config.n_samples].tolist()
            picked_states = [states[i] for i in idx_pick]
            # picked_states = states[idx_pick]
            picked_samples = env.state2oracle(picked_states)
            energies = oracle(picked_samples)
            # evaluate involves calculting distance between strings hence samples is sent
            gflownet.evaluate(
                picked_samples, energies, data_handler.train_dataset["samples"]
            )
            # dataset will eventuall store in proxy-format so states are sent to avoid the readable2state conversion
            data_handler.update_dataset(picked_states, energies)
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
