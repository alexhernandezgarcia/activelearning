#!/bin/bash
#SBATCH --job-name=activelearning
#SBATCH --ntasks=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output="/network/scratch/c/christina.humer/activelearning/runs/output-%j.txt"  # replace: location where you want to store the output of the job

module load anaconda/3 # replace: load anaconda module
conda activate al_new  # replace: conda env name
cd /home/mila/c/christina.humer/activelearning # replace: location of the code
python main.py --config-name test_branin user=chumer