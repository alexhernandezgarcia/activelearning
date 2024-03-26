"""
Runnable script for the active learning pipeline
"""

import sys
from dataset.dataset import BraninDatasetHandler, Branin_Data
from surrogate.surrogate import SingleTaskGPRegressor
from sampler.sampler import RandomSampler, GreedySampler
from filter.filter import OracleFilter, Filter
from gflownet.proxy.box.branin import Branin

import numpy as np
import hydra
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

n_iterations = 5  # TODO: replace with budget
grid_size = 10
float_prec = 64
n_samples = 3
train_path = "./storage/branin/data_%i_train.csv" % grid_size
maximize = False


@hydra.main(config_path="./config", config_name="main")
def main(config):
    # define candidate set
    xi = np.arange(0, grid_size)
    yi = np.arange(0, grid_size)
    grid = np.array(np.meshgrid(xi, yi))
    grid_flat = torch.tensor(grid.T, dtype=torch.float64).reshape(-1, 2)
    candidate_set, _ = Branin_Data(grid_size, grid_flat)[:]

    # Dataset
    dataset_handler = BraninDatasetHandler(
        grid_size=grid_size,
        train_path="./storage/branin/data_%i_train.csv" % grid_size,
        train_fraction=1.0,
        float_precision=float_prec,
    )
    # Oracle
    oracle = Branin(
        fidelity=1, do_domain_map=True, device=device, float_precision=float_prec
    )
    # Filter
    filter = Filter()

    for i in range(n_iterations):
        print("--iteration", i)
        train_data, test_data = dataset_handler.get_dataloader()
        # Surrogate (e.g., Bayesian Optimization)
        # starts with a clean slate each iteration
        surrogate = SingleTaskGPRegressor(
            float_precision=float_prec, device=device, maximize=maximize
        )
        surrogate.fit(train_data)

        # Sampler (e.g., GFlowNet, or Random Sampler)
        # also starts with a clean slate; TODO: experiment with NOT training from scratch
        # sampler = RandomSampler(surrogate)
        sampler = GreedySampler(surrogate)
        sampler.fit()  # only necessary for samplers that train a model

        samples = sampler.get_samples(
            n_samples * 3, candidate_set=candidate_set.clone().to(device)
        )
        filtered_samples = filter(
            n_samples=n_samples, candidate_set=samples.clone(), maximize=maximize
        )

        scores = oracle(filtered_samples.clone())
        dataset_handler.update_dataset(filtered_samples.cpu(), scores.cpu())

        print("Proposed Candidates:", filtered_samples)
        print("Oracle Scores:", scores)
        print("Best Score:", scores.min().cpu())


if __name__ == "__main__":
    main()
    sys.exit()
