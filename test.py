# Load Hydra config in notebooks
# https://github.com/facebookresearch/hydra/blob/main/examples/jupyter_notebooks/compose_configs_in_notebook.ipynb
import os
from hydra import initialize_config_dir, compose
import hydra
from omegaconf import OmegaConf

abs_config_dir = os.path.abspath("config/")

with initialize_config_dir(version_base=None, config_dir=abs_config_dir):
    config = compose(config_name="test_ocp.yaml", overrides=[])
    print(OmegaConf.to_yaml(config))
    print(config)

# config.sampler.conf.logger.do.online = False
import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = config.device
# device = "cuda"
n_iterations = config.budget  # TODO: replace with budget
n_samples = config.n_samples

from gflownet.utils.common import set_float_precision

float_prec = set_float_precision(config.float_precision)
# float_prec = set_float_precision(32)


from activelearning.utils.common import set_seeds

# Set other random seeds
# set_seeds(None)#config.seed)
ocp_checkpoint_path = config.dataset.checkpoint_path
dataset_path = config.dataset.data_path

from activelearning.dataset.ocp import OCPDatasetHandler
from activelearning.oracle.ocp import OCPOracle

from activelearning.surrogate.feature_extractor.mlp import MLP
from activelearning.surrogate.gp_surrogate import DeepKernelSVGPSurrogate
from activelearning.acquisition.acquisition import (
    BOTorchMaxValueEntropyAcquisition,
    BOTorchMonteCarloAcquisition,
)
import botorch

from activelearning.sampler.sampler import GreedySampler, RandomSampler
from activelearning.selector.selector import Selector, ScoreSelector
from gflownet.envs.crystals.surface import CrystalSurface
from functools import partial

# --- Environment
# env_maker = partial(CrystalSurface, atomgraphconverter=hydra.utils.instantiate(config.env.atomgraphconverter))
env_maker = hydra.utils.instantiate(config.env, _partial_=True)

# --- Dataset
dataset_handler = OCPDatasetHandler(
    env=env_maker(),
    checkpoint_path=ocp_checkpoint_path,
    data_path=dataset_path,
    train_fraction=0.1,
    float_precision=float_prec,
)

from activelearning.utils.logger import WandBLogger, ConsoleLogger

# logger = WandBLogger(project_name="test_ocp_training", run_name="greedy_epochs-20_samples-100_ei")
logger = ConsoleLogger(
    project_name="test_ocp_training", run_name="greedy_epochs-10_samples-100_ei"
)
# logger = None

from activelearning.utils.plotter import OCPCIME4RExportHelper

candidate_set, _, _ = dataset_handler.get_candidate_set()


# --- Oracle
oracle = OCPOracle(
    ocp_checkpoint_path,
    device=device,
    float_precision=float_prec,
)
best_scores = []
all_scores = {}


train_data, test_data = dataset_handler.get_dataloader()

# --- Surrogate (e.g., Bayesian Optimization, DKL)
# feature_extractor = Identity(dataset_handler.train_data.shape[-1])
feature_extractor = MLP(
    n_input=dataset_handler.train_data.shape[-1],
    n_hidden=[265, 512, 265],
    n_output=16,
    float_precision=float_prec,
)
feature_extractor.to(device)

# starts with a clean slate each iteration
surrogate = DeepKernelSVGPSurrogate(
    feature_extractor,
    float_precision=float_prec,
    device=device,
    mll_args={"num_data": len(train_data.dataset)},
    train_epochs=1,
    lr=0.01,
    logger=logger,
)
surrogate.fit(train_data)

# acq_fn = BOTorchMaxValueEntropyAcquisition(
#     surrogate.model, device=device, float_precision=float_prec
# )
acq_fn = BOTorchMonteCarloAcquisition(
    surrogate.model,
    acq_fn_class=botorch.acquisition.monte_carlo.qExpectedImprovement,
    dataset_handler=dataset_handler,
    device=device,
    float_precision=float_prec,
)
print("--- sampler")
# --- Sampler (e.g., GFlowNet, or Random Sampler)
# also starts with a clean slate; TODO: experiment with NOT training from scratch
# sampler = RandomSampler(acq_fn, device=device)
# sampler = GreedySampler(
#     acq_fn,
#     device=device,
#     float_precision=float_prec,
# )


sampler = hydra.utils.instantiate(
    config.sampler,
    dataset_handler=dataset_handler,
    env_maker=env_maker,
    acquisition=acq_fn,
    device=device,
    float_precision=float_prec,
    _recursive_=False,
)


sampler.fit()

samples, sample_idcs = sampler.get_samples(n_samples * 5, candidate_set=candidate_set)
