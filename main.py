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

    # Print working and logging directory
    print(f"Working directory of this run: {os.getcwd()}")
    print(
        "Logging directory of this run: "
        f"{hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )

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

    # --- Dataset
    dataset_handler = hydra.utils.instantiate(
        config.dataset,
        float_precision=config.float_precision,
    )
    candidate_set, _, _ = dataset_handler.get_candidate_set()

    # --- Oracle
    oracle = hydra.utils.instantiate(
        config.oracle,
        device=config.device,
        float_precision=config.float_precision,
    )

    best_scores = []
    for i in range(config.budget):
        print("--iteration", i)
        train_data, test_data = dataset_handler.get_dataloader()
        # --- Surrogate (e.g., Bayesian Optimization)
        # starts with a clean slate each iteration
        surrogate = hydra.utils.instantiate(
            config.surrogate,
            device=config.device,
            float_precision=config.float_precision,
            maximize=maximize,
        )
        surrogate.fit(train_data)

        # --- Acquisition
        # starts with a clean slate each iteration
        acquisition = hydra.utils.instantiate(
            config.acquisition,
            surrogate.model,
            dataset_handler=dataset_handler,
            device=config.device,
            float_precision=config.float_precision,
            maximize=maximize,
        )

        # --- Sampler (e.g., GFlowNet, or Random Sampler)
        # also starts with a clean slate; TODO: experiment with NOT training from scratch
        sampler = hydra.utils.instantiate(
            config.sampler,
            acquisition=acquisition,
            device=config.device,
            float_precision=config.float_precision,
            _recursive_=False,
        )
        sampler.fit()  # only necessary for samplers that train a model

        samples, sample_indices = sampler.get_samples(
            n_samples * 5, candidate_set=candidate_set
        )

        # --- Selector
        selector = hydra.utils.instantiate(
            config.selector,
            score_fn=acquisition.get_acquisition_values,
            device=config.device,
            float_precision=config.float_precision,
        )
        samples_selected, selected_idcs = selector(
            n_samples=n_samples, candidate_set=samples, index_set=sample_indices
        )

        oracle_samples = dataset_handler.prepare_dataset_for_oracle(
            samples_selected, selected_idcs
        )
        scores = oracle(oracle_samples)
        dataset_handler.update_dataset(samples_selected.cpu(), scores.cpu())

        print("Proposed Candidates:", samples_selected)
        print("Oracle Scores:", scores)
        print("Best Score:", scores.min().cpu())
        best_scores.append(scores.min().cpu())

    print("Best Scores:", best_scores)


if __name__ == "__main__":
    main()
    sys.exit()
