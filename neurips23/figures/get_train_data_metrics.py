"""
This script plots the topK energy with respective to the cumulative cost.
"""
import itertools
import random
import sys
from pathlib import Path

import biotite.sequence as biotite_seq
import biotite.sequence.align as align
import hydra
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
import selfies as sf
import torch
import wandb
import yaml
from diameter_clustering import LeaderClustering
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.SimDivFilters import rdSimDivPickers

from utils import get_hue_palette, get_pkl, plot_setup, get_dash, get_performance, get_diversity, get_n_modes


def build_dataframe(config):
    df = pd.DataFrame(
        columns=[
            "task",
            "al_type",
            "energy",
            "diversity",
            "n_modes",
            "k",
        ]
    )
    for task in ["branin", "hartmann", "dna", "amp", "molecules_ea", "molecules_ip"]:
        if task == "dna":
            substitution_matrix = align.SubstitutionMatrix.std_nucleotide_matrix()
        elif task == "amp":
            substitution_matrix = align.SubstitutionMatrix.std_protein_matrix()
        else:
            substitution_matrix = None
        with open(Path(f"config/io/{task}.yaml"), 'r') as f:
            config_task = yaml.safe_load(f)
        for al_type in ["sf", "mf"]:
            for k in config_task["data"]["k"]:
                train_data_f = (
                    Path(config.root_logdir)
                    / task
                    / "dataset"
                    / al_type
                    / "data_train.csv"
                )
                df_tr = pd.read_csv(train_data_f)
                if len(df_tr) < k:
                    continue
                if "energies" in df_tr:
                    if config_task["data"]["higherbetter"]:
                        idx_topk = np.argsort(df_tr.energies.values)[::-1][:k]
                    else:
                        idx_topk = np.argsort(df_tr.energies.values)[:k]
                    samples_topk = df_tr.samples.values[idx_topk]
                    energy_topk = np.mean(df_tr.energies.values[idx_topk])
                else:
                    energy_topk = None
                if config_task["data"]["do_diversity"]:
                    diversity_topk = get_diversity(samples_topk, task, substitution_matrix)
                    n_modes_topk = get_n_modes(
                        samples_topk,
                        task,
                        substitution_matrix,
                    )
                else:
                    diversity_topk = None
                    n_modes_topk = None
                df_aux = pd.DataFrame.from_dict(
                    {
                        "task": [task],
                        "al_type": [al_type],
                        "energy": [energy_topk],
                        "diversity": [diversity_topk],
                        "n_modes": [n_modes_topk],
                        "k": [k],
                    }
                )
                df = pd.concat([df, df_aux], axis=0, ignore_index=True)
        df.to_csv("data/train_stats.csv", index_label="index")
    return df



@hydra.main(config_path="./config", config_name="main", version_base=None)
def main(config):
    df = build_dataframe(config)


if __name__ == "__main__":
    main()
    sys.exit()
