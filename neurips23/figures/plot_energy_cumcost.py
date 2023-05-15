"""
This script plots the topK energy with respective to the cumulative cost.
"""
import itertools
import sys
from pathlib import Path

import biotite.sequence as biotite_seq
import biotite.sequence.align as align
import hydra
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import wandb
import yaml
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from utils import get_hue_palette, get_pkl, plot_setup


def build_dataframe(config):
    if config.io.task == "dna":
        substitution_matrix = align.SubstitutionMatrix.std_nucleotide_matrix()
    elif config.io.task == "amp":
        substitution_matrix = align.SubstitutionMatrix.std_protein_matrix()
    else:
        substitution_matrix = None
    df = pd.DataFrame(
        columns=["method", "seed", "energy", "cost", "diversity", "round", "k"]
    )
    for method in config.io.data.methods:
        for seed in config.io.data.methods[method].seeds:
            for k in config.io.data.k:
                logdir = (
                    Path(config.root_logdir)
                    / config.io.data.methods[method].dirname
                    / config.io.data.methods[method].seeds[seed].logdir
                )
                runpath = get_wandb_runpath(logdir)
                energy, cost, diversity = get_performance(
                    logdir,
                    runpath,
                    k,
                    config.io.data.higherbetter,
                    config.io.data.batch_size_al,
                    config.io.data.do_diversity,
                    config.io.task,
                    substitution_matrix,
                )
                n_rounds = len(energy)
                df_aux = pd.DataFrame.from_dict(
                    {
                        "method": [method for _ in range(n_rounds)],
                        "seed": [seed for _ in range(n_rounds)],
                        "energy": energy,
                        "cost": cost,
                        "diversity": diversity,
                        "round": np.arange(len(energy)),
                        "k": [k for _ in range(n_rounds)],
                    }
                )
                df = pd.concat([df, df_aux], axis=0, ignore_index=True)
    if "output_csv" in config.io:
        df.to_csv(config.io.output_csv, index_label="index")
    return df


def get_performance(
    logdir,
    runpath,
    k,
    higherbetter,
    batch_size,
    do_diversity=False,
    task=None,
    substitution_matrix=None,
):
    # Read data from experiment
    f_pkl = get_pkl(logdir)
    data_dict = pd.read_pickle(f_pkl)
    cumul_samples = data_dict["cumulative_sampled_samples"]
    cumul_energies = data_dict["cumulative_sampled_energies"]
    # Read data from wandb run
    api = wandb.Api()
    run = api.run(runpath)
    post_al_cum_cost = run.history(keys=["post_al_cum_cost"])
    post_al_cum_cost = np.unique(post_al_cum_cost["post_al_cum_cost"])
    # Compute metrics from each AL round
    rounds = np.arange(
        start=batch_size, stop=len(cumul_samples), step=batch_size, dtype=int
    )
    energy = []
    cost = []
    diversity = []
    for idx, upper_bound in enumerate(rounds):
        # Compute mean topk energy up to current round
        cumul_sampled_energies_curr_round = cumul_energies[:upper_bound].cpu().numpy()
        if higherbetter:
            idx_topk = np.argsort(cumul_sampled_energies_curr_round)[::-1][:k]
        else:
            idx_topk = np.argsort(cumul_sampled_energies_curr_round)[:k]
        energies_topk = cumul_sampled_energies_curr_round[idx_topk]
        mean_energy_topk = np.mean(energies_topk)
        # Compute diversity of topk samples, if requested
        if do_diversity and k > 1:
            cumul_samples_curr_round = np.array(cumul_samples[:upper_bound])
            samples_topk = cumul_samples_curr_round[idx_topk]
            mean_diversity_topk = get_diversity(samples_topk, task, substitution_matrix)
        # Append to lists
        energy.append(mean_energy_topk)
        cost.append(post_al_cum_cost[idx])
        if do_diversity and k > 1:
            diversity.append(mean_diversity_topk)
    if not do_diversity or k == 1:
        diversity = [None for _ in range(len(energy))]
    return energy, cost, diversity


def get_diversity(seqs, task=None, substitution_matrix=None):
    if task == "dna":
        seqs = [biotite_seq.NucleotideSequence(seq) for seq in seqs]
        distances = []
        for pair in itertools.combinations(seqs, 2):
            alignment = align.align_optimal(
                pair[0], pair[1], substitution_matrix, local=False, max_number=1
            )[0]
            distances.append(align.get_sequence_identity(alignment))
    elif task == "amp":
        seqs = [biotite_seq.ProteinSequence(seq) for seq in seqs]
        distances = []
        for pair in itertools.combinations(seqs, 2):
            alignment = align.align_optimal(
                pair[0], pair[1], substitution_matrix, local=False, max_number=1
            )[0]
            distances.append(align.get_sequence_identity(alignment))
    else:
        sample_states1 = torch.tensor(seqs)
        sample_states2 = sample_states1.clone()
        dist_matrix = torch.cdist(sample_states1, sample_states2, p=2)
        dist_upper_triangle = torch.triu(dist_matrix, diagonal=1)
        distances = dist_upper_triangle[dist_upper_triangle != 0]
        distances = distances.numpy()
    return np.mean(distances)


def process_cost(df, config):
    if config.plot.x_axis.type == "fraction_budget":
        df.cost = df.cost / df.cost.max()
    return df


def get_wandb_runpath(logdir):
    with open(Path(logdir) / "wandb.url", "r") as f:
        url = f.read()
    entity, project, _, run_id = (
        url.rstrip().replace("https://wandb.ai/", "").split("/")
    )
    runpath = entity + "/" + project + "/" + run_id
    runpath = runpath.replace("%20", " ")
    return runpath


def plot(df, config):
    plot_setup()

    fig, ax = plt.subplots(
        figsize=(config.plot.width, config.plot.height), dpi=config.plot.dpi
    )

    # Plot
    sns.lineplot(ax=ax, data=df, x="cost", y="energy", hue="method", style="k")

    # Set X-label
    ax.set_xlabel(config.plot.x_axis.label)
    # Set Y-label
    if config.io.data.higherbetter:
        better = "higher"
    else:
        better = "lower"
    ax.set_ylabel(f"Top-{config.io.data.k} energy ({better} is better)")

    # Legend
    leg_handles, _ = ax.get_legend_handles_labels()
    leg_labels = [
        config.io.data.methods[method].name for method in config.io.data.methods
    ]
    leg = ax.legend(
        handles=leg_handles,
        labels=leg_labels,
        loc="best",
        title="",
        framealpha=1.0,
        frameon=True,
    )

    # Change spines
    sns.despine(ax=ax, left=True, bottom=True)

    return fig


@hydra.main(config_path="./config", config_name="main", version_base=None)
def main(config):
    # Determine output dir
    if config.io.output_dir.upper() == "SLURM_TMPDIR":
        output_dir = Path(os.environ["SLURM_TMPDIR"])
    else:
        output_dir = Path(to_absolute_path(config.io.output_dir))
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
    # Store args
    output_yml = output_dir / Path(Path(config.io.output_filename).stem + ".yaml")
    with open(output_yml, "w") as fp:
        OmegaConf.save(config=config, f=fp.name)
    # Build data frame or read CSV
    if (
        "input_csv" in config.io
        and config.io.input_csv is not None
        and Path(config.io.input_csv).exists()
    ):
        df = pd.read_csv(config.io.input_csv, index_col="index")
    else:
        df = build_dataframe(config)
    df = process_cost(df, config)
    # Plot
    fig = plot(df, config)
    # Save figure
    output_fig = output_dir / config.io.output_filename
    fig.savefig(output_fig, bbox_inches="tight")


if __name__ == "__main__":
    main()
    sys.exit()