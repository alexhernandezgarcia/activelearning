"""
Runnable script for the active learning pipeline
"""

import os
import random
import sys

import hydra

from activelearning.utils.common import set_seeds


@hydra.main(config_path="./config", config_name="main", version_base="1.1")
def main(config):

    # Get current directory and set it as root log dir for Logger
    cwd = os.getcwd()
    config.logger.logdir.root = cwd
    print(f"\nLogging directory of this run:  {cwd}\n")

    # Reset seed for job-name generation in multirun jobs
    random.seed(None)
    # Set other random seeds
    set_seeds(config.seed)

    # Logger
    logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)

    # Active learning variables
    # TODO: rethink where this configuration should go
    n_samples = config.n_samples
    maximize = config.maximize

    # Dataset
    dataset_handler = hydra.utils.instantiate(
        config.dataset,
        float_precision=config.float_precision,
    )
    candidate_set, xi, yi = dataset_handler.get_candidate_set()

    # Oracle
    oracle = hydra.utils.instantiate(
        config.oracle,
        device=config.device,
        float_precision=config.float_precision,
    )
    # Filter
    filter = hydra.utils.instantiate(
        config.filter,
        oracle=oracle,
    )

    best_scores = []
    for i in range(config.budget):
        print("--iteration", i)
        train_data, test_data = dataset_handler.get_dataloader()
        # Surrogate (e.g., Bayesian Optimization)
        # starts with a clean slate each iteration
        surrogate = hydra.utils.instantiate(
            config.surrogate,
            device=config.device,
            float_precision=config.float_precision,
            maximize=maximize,
        )
        surrogate.fit(train_data)

        # Sampler (e.g., GFlowNet, or Random Sampler)
        # also starts with a clean slate; TODO: experiment with NOT training from scratch
        sampler = hydra.utils.instantiate(
            config.sampler,
            surrogate=surrogate,
            device=config.device,
            float_precision=config.float_precision,
            _recursive_=False,
        )
        sampler.fit()  # only necessary for samplers that train a model

        samples = sampler.get_samples(
            n_samples * 3, candidate_set=candidate_set.clone().to(device)
        )
        filtered_samples = filter(n_samples=n_samples, candidate_set=samples.clone())

        scores = oracle(filtered_samples.clone())
        dataset_handler.update_dataset(filtered_samples.cpu(), scores.cpu())

        print("Proposed Candidates:", filtered_samples)
        print("Oracle Scores:", scores)
        print("Best Score:", scores.min().cpu())
        best_scores.append(scores.min().cpu())

    print("Best Scores:", best_scores)


if __name__ == "__main__":
    main()
    sys.exit()
