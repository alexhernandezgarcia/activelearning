"""
Runnable script for the active learning pipeline
"""
import sys
from dataset.dataset import BraninDatasetHandler, Branin_Data
from surrogate.surrogate import SingleTaskGPRegressor
from sampler.sampler import RandomSampler
from filter.filter import OracleFilter
from gflownet.proxy.box.branin import Branin

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
import numpy as np

n_iterations = 5 # TODO: replace with budget
grid_size = 10
float_prec = 64
n_samples = 5
train_path = "./storage/branin/data_%i_train.csv"%grid_size

def main():
    # define candidate set
    xi = np.arange(0,grid_size)
    yi = np.arange(0,grid_size)
    grid = np.array(np.meshgrid(xi,yi))
    grid_flat = torch.tensor(grid.T, dtype=torch.float64).reshape(-1,2).to(device)
    candidate_set, _ = Branin_Data(grid_flat, grid_size=grid_size, device=device)[:]
    
    
    # Dataset
    dataset_handler = BraninDatasetHandler(train_path=train_path, train_fraction=1.0, device=device, float_precision=float_prec)
    # Oracle
    oracle = Branin(fidelity=1, do_domain_map=True, device=device, float_precision=float_prec)
    # Filter
    filter = OracleFilter(oracle)


    for i in range(n_iterations):
        print("--iteration", i)
        # Surrogate (e.g., Bayesian Optimization)
        # starts with a clean slate each iteration
        surrogate = SingleTaskGPRegressor(float_precision=float_prec, device=device)
        surrogate.fit(dataset_handler.train_data)
        
        # Sampler (e.g., GFlowNet, or Random Sampler)
        # also starts with a clean slate; TODO: experiment with NOT training from scratch
        sampler = RandomSampler(surrogate)
        sampler.fit() # only necessary for samplers that train a model

        samples = sampler.get_samples(n_samples*3, candidate_set=candidate_set.clone())
        filtered_samples = filter(n_samples=n_samples, candidate_set=samples.clone(), maximize=False)
        
        scores = oracle(filtered_samples.clone())
        dataset_handler.update_dataset(filtered_samples, scores)

        print("Proposed Candidates:", filtered_samples)
        print("Oracle Scores:", scores)



if __name__ == "__main__":
    main()
    sys.exit()