# Multi-Fidelity Active Learning with GFlowNets

This repository contains code for the paper Multi-Fidelity Active Learning with GFLowNets. 

## Abstract

In the last decades, the capacity to generate large amounts of data in science and
engineering applications has been growing steadily. Meanwhile, the progress in
machine learning has turned it into a suitable tool to process and utilise the available
data. Nonetheless, many relevant scientific and engineering problems present
challenges where current machine learning methods cannot yet efficiently leverage
the available data and resources. For example, in scientific discovery, we are often
faced with the problem of exploring very large, high-dimensional spaces, where
querying a high fidelity, black-box objective function is very expensive. Progress
in machine learning methods that can efficiently tackle such problems would help
accelerate currently crucial areas such as drug and materials discovery. In this paper,
we propose the use of GFlowNets for multi-fidelity active learning, where multiple
approximations of the black-box function are available at lower fidelity and cost.
GFlowNets are recently proposed methods for amortised probabilistic inference
that have proven efficient for exploring large, high-dimensional spaces and can
hence be practical in the multi-fidelity setting too. Here, we describe our algorithm
for multi-fidelity active learning with GFlowNets and evaluate its performance in
both well-studied synthetic tasks and practically relevant applications of molecular
discovery. Our results show that multi-fidelity active learning with GFlowNets
can efficiently leverage the availability of multiple oracles with different costs and
fidelities to accelerate scientific discovery and engineering design.

![Overview of the algorithm](http://url/to/img.png)

## Key Results 
We evaluate the proposed MF-GFN approach in both synthetic tasks (Branin, Hartmann) and benchmark tasks of practical relevance, such as DNA
aptamer generation, antimicrobial peptide design and molecular modelling (read section 4 of the paper). Through comparisons
with previously proposed methods as well as with variants of our method designed to understand the
contributions of different components, we conclude that multi-fidelity active learning with GFlowNets
not only outperforms its single-fidelity active learning counterpart in terms of cost effectiveness and
diversity of sampled candidates, but it also offers an advantage over other multi-fidelity methods due
to its ability to learn a stochastic policy to jointly sample objects and the fidelity of the oracle to be
used to evaluate them.

![Results on AMP, DNA and Molecules (IP)](http://url/to/img.png)

## Setup

1. Install the GFlowNet codebase compatible with this implementation from [alexhernandezgarcia/gflownet](https://github.com/alexhernandezgarcia/gflownet/tree/mfgfn-v1.0).
``` 
python -m pip install --upgrade https://github.com/alexhernandezgarcia/gflownet/archive/cont_mf.zip 
```
2.  Clone this repository and install other dependencies by running ```pip install -r requirements.txt``` where this repository is cloned.
3. Set up the AMP oracles (optional, required only if you wish to run experiments with the anti-microbial peptide environment). Install the clamp-common-eval library from [MJ10/clamp-gen-data](https://github.com/MJ10/clamp-gen-data/tree/mfgfn-v1.0) by cloning the repo and then running the following where the repository is cloned: 
```
pip install -r requirements.txt && pip install -e .
``` 
4. Rename the log directory and data directory arguments (if necessary) in `config/user/anonymous.yaml` 

## Usage
The project uses [Hydra](https://hydra.cc/) for configuration and [Weights and Biases](https://docs.wandb.ai/) for logging.
For reproducing the results, configuration files with the default settings for experiments with the synthetic and benchmark tasks have been created in `config/`. Run
```
python activelearning.py --config_name=<config-filename>
```

### Default Config Options
Options `sf` and `mf` for the `<fid>` placeholder correspond to the single-fidelity and multi-fidelity variants respectively. Files prefixed with `ppo` train the PPO (instead of the GFlowNet) algorithm as the sampler.

- `<fid>_branin`: Branin 
- `<fid>_hartmann`: Hartmann 
- `<fid>_amp`: Antimicrobial Peptides
- `<fid>_aptamers`: DNA
- `<fid>_mols_ea`: Molecules with the objective of maximizing electron affinity 
- `<fid>_mols_ip`: Molecules with the objective of maximizing (negative) ionization potential  

### Additional Config Options
Below we list other configuration options. See the config files in `./config` for all configurable parameters. Note that any config field can be overridden from the command line, and some configurations are not supported.

**Environment Options**
- `grid` (for the synthetic tasks, Branin and Hartmann)
- `aptamers`  (for DNA)
- `amp` (for antimicrobial peptides)
- `mols` (for molecules represented as SELFIES strings)

**Oracle Options**

The oracles are prefixed by the environment (`branin`, `hartmann`, `amp`, `aptamers`, `mols_ea`, `mols_ip`) and indexed by increasing level of fidelity. 

**Sampler Options**
- `gflownet` (gflownet with the flowmatch objective)
- `trajectorybalance` (gflownet with the trajectory balance objective, recommended for reproducing results)
- `ppo` (RL baseline, proximal policy optimization algorithm)

**Proxy Options**

Options `sf` and `mf` for the `<fid>` placeholder correspond to the single-fidelity and multi-fidelity variants repectively.
- `<fid>_gp` (exact gaussian process for regression)
- `<fid>_svgp` (stochastic vartiation gaussian process for regression)
- `<fid>_dkl` (deep kernel regressor with backbone as one of the model options and index kernel for learning the fidelity parameter)
- `mf_dkl_linear` (multi-fidelity deep kernel regressor with backbone as one of the model options (below) and linear downsampling kernel for learning the fidelity parameter)

**Model Options** 
- `mlp`
- `transformer`
- `mlm_cnn` (masked language model based on CNN layers)
- `mlm_transformer` (masked language model based transformer)
- `regressive` (based on the [DNN-MFBO implementation](https://github.com/shib0li/DNN-MFBO))
- `mf_mlp` (MLP with an additional layer over the concatenated representation of the feature with fidelity)


**Acquisition Function Options**

Single and multi-fidelity variants of the Max Entropy Search (MES) acquisiton function have been implemented. The suffix (`<gp>`, `<svgp>`, `<dkl>` indicate with which regressor the implementation is compatible with.)
<!-- 
## Citation
If you use any part of this code for your own work, please cite -->
