"""
Runnable script for the active learning pipeline
"""

import os
import random
import sys

import hydra

from activelearning.utils.common import set_seeds
import torch


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
    # logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)
    logger = hydra.utils.instantiate(config.logger, conf=config, _recursive_=False)

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
    all_scores = {}
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
            logger=logger,
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
        scores = oracle(oracle_samples).cpu()
        dataset_handler.update_dataset(oracle_samples.cpu(), scores)

        print("Proposed Candidates:", samples_selected)
        print("Oracle Scores:", scores)
        print("Best Score:", scores.min().cpu())
        best_scores.append(scores.min().cpu())
        all_scores[i] = scores.tolist()
        if logger is not None:
            logger.log_metric(scores.min(), "best_score")
            logger.log_metric(torch.median(scores), "median_score")
            logger.log_metric(torch.mean(scores), "mean_score")
            logger.log_metric(scores.max(), "worst_score")
            logger.log_metric(scores, "scores")

            logger.log_step(i)

    if logger is not None:
        import matplotlib.pyplot as plt

        plt.boxplot(all_scores.values(), labels=all_scores.keys())
        plt.ylim(top=50, bottom=-50)
        logger.log_figure(plt, "all_scores")
    print("Best Scores:", best_scores)


if __name__ == "__main__":
    main()
    sys.exit()
