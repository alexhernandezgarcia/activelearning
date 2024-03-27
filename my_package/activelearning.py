"""
Runnable script for the active learning pipeline
"""

import sys
from dataset.dataset import BraninDatasetHandler, Branin_Data
from surrogate.surrogate import SingleTaskGPRegressor
from sampler.sampler import RandomSampler, GreedySampler
from filter.filter import OracleFilter, Filter
from gflownet.proxy.box.branin import Branin
from utils.plotter import PlotHelper
import matplotlib.colors as cm
import matplotlib.pyplot as plt

colors = plt.get_cmap("Reds")

import numpy as np
import hydra
import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(config_path="./config", config_name="main", version_base="1.1")
def main(config):

    # init config
    device = config.device
    n_iterations = config.budget  # TODO: replace with budget
    float_precision = config.float_precision
    n_samples = config.n_samples
    maximize = config.maximize

    # Logger
    logger = hydra.utils.instantiate(
        config.logger,
        config,
        _recursive_=False,
    )

    # Plotter
    plotter = PlotHelper(logger)

    # Dataset
    dataset_handler = hydra.utils.instantiate(
        config.dataset,
        float_precision=config.float_precision,
    )
    candidate_set, xi, yi = dataset_handler.get_candidate_set()

    # Oracle
    oracle = hydra.utils.instantiate(
        config.oracle,
        device=device,
        float_precision=float_precision,
    )
    # Filter
    filter = hydra.utils.instantiate(
        config.filter,
        oracle=oracle,
    )

    if plotter is not None:
        fig_oracle, ax_oracle = plotter.plot_function(
            oracle, candidate_set.clone().to(device), xi=xi, yi=yi
        )

    best_scores = []
    for i in range(n_iterations):
        print("--iteration", i)
        train_data, test_data = dataset_handler.get_dataloader()
        # Surrogate (e.g., Bayesian Optimization)
        # starts with a clean slate each iteration
        surrogate = hydra.utils.instantiate(
            config.surrogate,
            device=config.device,
            float_precision=float_precision,
            maximize=maximize,
        )
        surrogate.fit(train_data)

        # Sampler (e.g., GFlowNet, or Random Sampler)
        # also starts with a clean slate; TODO: experiment with NOT training from scratch
        sampler = hydra.utils.instantiate(
            config.sampler,
            surrogate=surrogate,
            logger=logger,
            device=device,
            float_precision=float_precision,
            _recursive_=False,
        )
        sampler.fit()  # only necessary for samplers that train a model

        samples = sampler.get_samples(
            n_samples * 3, candidate_set=candidate_set.clone().to(device)
        )
        filtered_samples = filter(
            n_samples=n_samples, candidate_set=samples.clone(), maximize=maximize
        )

        if plotter is not None:
            fig_acq, ax_acq = plotter.plot_function(
                surrogate, candidate_set.clone().to(device), xi=xi, yi=yi
            )
            fig_acq, ax_acq = plotter.plot_samples(filtered_samples, ax_acq, fig_acq)
            ax_acq.set_title("acquisition fn + selected samples of iteration %i" % i)
            plotter.log_figure(fig_acq, "acq")

        if ax_oracle is not None:
            ax_oracle.scatter(
                x=filtered_samples[:, 0].cpu(),
                y=filtered_samples[:, 1].cpu(),
                c=cm.to_hex(colors(i / n_iterations)),
                marker="x",
                label="it %i" % i,
            )

        scores = oracle(filtered_samples.clone())
        dataset_handler.update_dataset(filtered_samples.cpu(), scores.cpu())

        print("Proposed Candidates:", filtered_samples)
        print("Oracle Scores:", scores)
        print("Best Score:", scores.min().cpu())
        best_scores.append(scores.min().cpu())

    if plotter is not None:
        fig_oracle.legend()
        ax_oracle.set_title("oracle fn + samples")
        plotter.log_figure(fig_oracle, key="oracle")

    fig = plt.figure()
    plt.plot(best_scores)
    plt.xlabel("iterations")
    plt.ylabel("scores")
    plt.title("Best Score in each iteration")
    if plotter is not None:
        plotter.log_figure(fig, key="best_scores")


if __name__ == "__main__":
    main()
    sys.exit()
