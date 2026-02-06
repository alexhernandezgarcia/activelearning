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

    # Active learning variables
    # TODO: rethink where this configuration should go
    n_samples = config.n_samples

    # --- Environment
    env_maker = hydra.utils.instantiate(
        config.env,
        device=config.device,
        float_precision=config.float_precision,
        _partial_=True,
    )

    # --- Dataset
    dataset_handler = hydra.utils.instantiate(
        config.dataset,
        env=env_maker(),
        float_precision=config.float_precision,
    )

    plotter = None
    if "plotter" in config:
        plotter = hydra.utils.instantiate(
            config.plotter, dataset_handler=dataset_handler, device=config.device
        )

    candidate_set, _, _ = dataset_handler.get_candidate_set()

    # --- Oracle
    oracle = hydra.utils.instantiate(
        config.oracle,
        device=config.device,
        float_precision=config.float_precision,
    )

    if plotter is not None:
        plot_set, _, _ = dataset_handler.get_candidate_set(as_dataloader=False)
        oracle_dataloader = dataset_handler.prepare_oracle_dataloader(plot_set)
        plotter.plot_function(
            oracle,
            oracle_dataloader,
            label="oraclefn",
            fig="",
            ax="",
        )
    # Logger
    run_name = config.logger.run_name
    # logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)

    best_scores = []
    all_scores = {}
    for i in range(config.budget):
        train_data, test_data = dataset_handler.get_dataloader()
        print("--iteration", i, "- training on", len(train_data.dataset), "instances")
        # Logger
        # logger.set_context(i)
        # TODO: initializing the logger every iteration is not ideal; is it possible in wandb to do this in a smarter way? (i.e., only have one run of wandb that logs the active learning loop and all subsequent model training + metrics?)
        config.logger.run_name = run_name + "_it-%i" % i
        logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)
        # --- Surrogate (e.g., Bayesian Optimization)
        # starts with a clean slate each iteration
        if (
            config.surrogate.get("mll_args") is not None
            and config.surrogate.mll_args.get("num_data") is not None
        ):
            config.surrogate.mll_args.num_data = len(train_data.dataset)
        surrogate = hydra.utils.instantiate(
            config.surrogate,
            device=config.device,
            float_precision=config.float_precision,
            logger=logger,
        )
        surrogate.fit(train_data, step=i)

        if plotter is not None:
            plotter.plot_function(
                surrogate.get_predictions,
                candidate_set,
                output_index=0,
                fig="",
                ax="",
                label="pred_target_mean",
                iteration=i,
            )
            plotter.plot_function(
                surrogate.get_predictions,
                candidate_set,
                output_index=1,
                fig="",
                ax="",
                label="pred_target_var",
                iteration=i,
            )

        # --- Acquisition
        # starts with a clean slate each iteration
        acquisition = hydra.utils.instantiate(
            config.acquisition,
            surrogate.get_model(),
            dataset_handler=dataset_handler,
            device=config.device,
            float_precision=config.float_precision,
        )
        if plotter is not None:
            plotter.plot_function(
                acquisition,
                candidate_set,
                fig="",
                ax="",
                label="acq",
                iteration=i,
            )

        # --- Sampler (e.g., GFlowNet, or Random Sampler)
        # also starts with a clean slate; TODO: experiment with NOT training from scratch
        sampler = hydra.utils.instantiate(
            config.sampler,
            env_maker=env_maker,
            acquisition=acquisition,
            dataset_handler=dataset_handler,
            logger=logger,
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
            score_fn=acquisition,
            device=config.device,
            float_precision=config.float_precision,
            maximize=True,
        )
        samples_selected, selected_idcs = selector(
            n_samples=n_samples,
            candidate_set=dataset_handler.states2selector(samples),
            index_set=sample_indices,
        )

        oracle_samples = dataset_handler.states2oracle(samples_selected)
        scores = oracle(oracle_samples).cpu()
        scores = dataset_handler.update_dataset(samples_selected, scores)

        print("Proposed Candidates:", oracle_samples)
        print("Oracle Scores:", scores)
        print("Best Score:", scores.max())
        best_scores.append(scores.max())
        all_scores[i] = scores
        if logger is not None:
            scores_flat = torch.stack(list(all_scores.values())).flatten()
            mean_top_k = scores_flat.topk(n_samples, largest=True).values.mean()
            metrics = {
                "top_score": scores_flat.max(),
                "mean_topk_score": mean_top_k,
                "best_score": scores.max(),
                "median_score": torch.median(scores),
                "mean_score": torch.mean(scores),
                "worst_score": scores.min(),
                "scores": scores,
            }
            logger.log_metrics(metrics, use_context=False, step=None)

        if plotter is not None and selected_idcs is not None:
            plotter.plot_scores(selected_idcs, scores, i + 1)
        logger.end()

    # if logger is not None:
    #     import matplotlib.pyplot as plt

    #     plt.boxplot(all_scores.values(), labels=all_scores.keys())
    #     plt.ylim(top=50, bottom=-50)

    # if plotter is not None:
    #     plotter.end(filename=logger.run_name + ".csv")
    print("Best Scores:", best_scores)


if __name__ == "__main__":
    main()
    sys.exit()
