# Multi-Fidelity Active Learning with GFlowNets

This repository extends the code used in the paper [Multi-Fidelity Active Learning with GFLowNets](http://arxiv.org/abs/2306.11715), implemented in [github.com/nikita-0209/mf-al-gfn](https://github.com/nikita-0209/mf-al-gfn).

## Installation

+ This code **requires** `python 3.10`.
+ If you are installing this in a compute cluster (for example, Mila's), you can load the required modules by running `source ./prereq_cluster_cpu.sh`.
+ Setup is currently only supported on Ubuntu. It should also work on OSX, but you will need to handle the package dependencies.
+ The recommended installation is as follows:

```bash
python3.10 -m venv ~/venvs/activelearning  # Initalize your virtual env.
source ~/envs/activelearning/bin/activate  # Activate your environment.
./prereq_python.sh  # Updates pip and forces the installation of potentially problematic libraries
python -m pip install .[all]  # Install the remaining dependencies of this package.
```

The above steps install PyTorch for CPU only. In order to install a cuda-enabled PyTorch, it must use the wheels for cuda 11.8 in order to be compatible with the GFlowNet package. In the cluster, first run `source ./prereq_cluster_gpu.sh` and then install PyTorch:

```
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

Use the following commands to install FAENet:

```bash
pip install git+https://github.com/RolnickLab/ocp.git@uncertainty-depfaenet
```

For development you can use a local installation of the package:

```bash
git clone https://github.com/RolnickLab/ocp.git
cd ocp
pip install -e .
```

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
