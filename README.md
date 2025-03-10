# Multi-Fidelity Active Learning with GFlowNets

This repository extends the code used in the paper [Multi-Fidelity Active Learning with GFLowNets](http://arxiv.org/abs/2306.11715), implemented in [github.com/nikita-0209/mf-al-gfn](https://github.com/nikita-0209/mf-al-gfn).

## Installation

**If you simply want to install everything on a GPU-enabled machine, clone the repo and run `install.sh`:**

```bash
git clone git@github.com:alexhernandezgarcia/activelearning.git
cd activelearning
source install.sh
```

- This project **requires** Python 3.10 and CUDA 11.8.
- It is also **possible to install a CPU-only environment** that supports most features (see below).
- Setup is currently only supported on Ubuntu. It should also work on OSX, but you will need to handle the package dependencies.

### Step by step installation

The following steps, as well as the script `install.sh`, assume the use of Python virtual environments for the installation.

1. Ensure that you have Python 3.10 and, if you want to install GPU-enabled PyTorch, CUDA 11.8. In a cluster that uses [modules](https://hpc-wiki.info/hpc/Modules), you may be able to load Python and CUDA with:

```bash
module load python/3.10
module load cuda/11.8
```

2. Create and activate a Python virtual environment with `venv`. For example:

```bash
python -m venv activelearning-env
source activelearning-env/bin/activate
```

3. Install PyTorch 2.5.1.

For a CUDA-enabled installation:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
```

For a CPU-only installation:

```bash
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
```

4. Install the rest of the dependencies:

```bash
python -m pip install .
```

The above command will install the minimum set of dependencies to run the core features of the activelearning package. Specific features require the installation of extra dependencies. Currently, these are the existing sets of extras:

- `dev`: dependencies for development, such as linting and testing packages.
- `materials`: dependencies for materials applications, such as the Crystal-GFN.

Extras can be installed by specifying the tags in square brackets:

```bash
python -m pip install .[dev]
```

or

```bash
python -m pip install .[dev,materials]
```

### Installing with `install.sh`

The script `install.sh` simplifies the installation of a Python environment with the necessary or desired dependencies.

By default, running `source install.sh` will create a Python environment in `./activelearning-env with CUDA-enabled PyTorch and all the dependecies (all extras). However, the script admits the following arguments to modify the configuration of the environment:

- `--cpu`: Install CPU-only PyTorch (mutually exclusive with --cuda).
- `--cuda`: Install CUDA-enabled PyTorch (default, and mutually exclusive with --cpu).
- `--envpath PATH`: Path of the Python virtual environment to be installed. Default: `./activelearning-env`
- `--extras LIST`: Comma-separated list of extras to install. Default: `all`. Options:
    - dev: dependencies for development, such as linting and testing packages.
    - materials: dependencies for materials applications, such as the Crystal-GFN.
    - all: all of the above
    - minimal: none of the above, that is the minimal set of dependencies.
- `--dry-run`: Print the summary of the configuration selected and exit.
- `--help`: Show the help message and exit.

For example, you may run:

```bash
source install.sh --cpu --envpath ~/myenvs/activelearning-env --extras dev,materials
```

to install an environment on `~/myenvs/activelearning-env`, with a CPU-only PyTorch and the dev and materials extras.

## Run Examples
To run the different examples you can use the following command:

```bash
python main.py user=<user-filename> +tests=branin
```

### Pre-Defined Config Files
The following pre-defined config files are available:
- branin
- hartmann
- ocp

### Customize Config Options
Some config options can be customized independently of the task. Here is a list of possible options for each component:
- sampler: random | greedy | random_gflownet | gflownet
- selector: selector | score
- surrogate: gp | dkl | svdkl_kernel_wrapper
- acquisition: botorch_ei | botorch_mve | botorch_nei
- user: default | <custom_user_file>
- logger: wandb | base


## Citation
If you use this code for your own work, please consider citing our published work:
```
@misc{hernandezgarcia2023multifidelity,
      title={Multi-Fidelity Active Learning with GFlowNets}, 
      author={Alex Hernandez-Garcia and Nikita Saxena and Moksh Jain and Cheng-Hao Liu and Yoshua Bengio},
      year={2023},
      eprint={2306.11715},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
