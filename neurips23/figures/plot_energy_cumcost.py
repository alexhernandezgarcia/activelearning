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
import matplotlib.colors as mcolors
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
    # Catch cases where post_al_cum_cost has fewer values than number of rounds
    rounds = rounds[:len(post_al_cum_cost)]
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
    for method in df.method.unique():
        for k in df.k.unique():
            df_aux = df.loc[(df.method == method) & (df.k == k)]
            if len(df_aux.seed.unique()) > 1:
                costs_dict = {}
                for seed in df_aux.seed.unique():
                    costs_dict.update(
                        {seed: df_aux.loc[df_aux.seed == seed, "cost"].values}
                    )
                max_lengths = np.sort([len(c) for c in costs_dict.values()])
                idx_from = 0
                for idx_to in max_lengths:
                    costs_lists = [
                        c[idx_from:idx_to]
                        for c in costs_dict.values()
                        if len(c) > idx_from
                    ]
                    if len(costs_lists) <= 1:
                        break
                    costs_mean = np.mean(np.stack(costs_lists, axis=1), axis=1)
                    for seed, vec in costs_dict.items():
                        if len(vec) > idx_from:
                            costs_dict[seed][idx_from:idx_to] = costs_mean
                    idx_from = idx_to
                for seed in df_aux.seed.unique():
                    df.loc[
                        (df.method == method) & (df.k == k) & (df.seed == seed), "cost"
                    ] = costs_dict[seed]
    if config.plot.x_axis.type == "fraction_budget":
        df.cost = df.cost / df.cost.max()
    return df


def process_highlights(df, config):
    df["linewidth"] = np.ones(len(df))
    for method in config.io.data.methods:
        if config.io.data.methods[method].highlight:
            linewidth = config.plot.linewidth.highlight
        else:
            linewidth = config.plot.linewidth.other
        df.loc[df.method == method, "linewidth"] = linewidth
    return df


def process_diversity(df, config):
    min_diversity = df.diversity.min()
    max_diversity = df.diversity.max()
    df.diversity = 1.0 / df.diversity 
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


def make_palette(config):
    palette = {}
    for method in config.plot.colors:
        palette.update(
            {
                method: sns.color_palette(
                    config.plot.colors[method].palette, as_cmap=False, n_colors=9
                )[config.plot.colors[method].index]
            }
        )
    return palette


def plot(df, config):
    if config.io.data.higherbetter:
        opt = "Max."
        better = "higher"
    else:
        opt = "Min."
        better = "lower"

    plot_setup()

    fig, ax = plt.subplots(
        figsize=(config.plot.width, config.plot.height), dpi=config.plot.dpi
    )

    palette = make_palette(config)

    # Plot
    if config.plot.do_all_k:
        k_plot = "K"
        sns.lineplot(
            ax=ax,
            data=df,
            x="cost",
            y="energy",
            hue="method",
            style="k",
            estimator=config.plot.estimator,
            markers=config.plot.do_markers,
            palette=palette,
        )
        leg_handles_def, leg_labels_def = ax.get_legend_handles_labels()
    else:
        k_plot = df.k.max()
        sns.lineplot(
            ax=ax,
            data=df.loc[df.k == k_plot],
            x="cost",
            y="energy",
            hue="method",
            size="linewidth",
            estimator=config.plot.estimator,
            markers=config.plot.do_markers,
            palette=palette,
        )
        leg_handles_def, leg_labels_def = ax.get_legend_handles_labels()
        # Scatter plot of diversity
        df_means = df.groupby(
            ["round", "method", "k"], group_keys=False, as_index=False
        ).mean()
        gray_palette = mcolors.Colormap("Grays")
        sns.scatterplot(
            ax=ax,
            data=df_means.loc[df_means.k == k_plot],
            x="cost",
            y="energy",
            hue="diversity",
#             sizes=(10.0, 100.0),
            palette="gist_gray",
            zorder=10,
        )
#         sns.scatterplot(
#             ax=ax,
#             data=df_means.loc[df_means.k == k_plot],
#             x="cost",
#             y="energy",
#             hue="method",
#             size="diversity",
#             sizes=(10.0, 100.0),
#             palette=palette,
#         )
    #         sns.lineplot(
    #             ax=ax,
    #             data=df.loc[df.k == k_plot],
    #             x="cost",
    #             y="energy",
    #             hue="method",
    #             style="method",
    #             estimator=config.plot.estimator,
    #             markers=True,
    #         )
    #         sns.lineplot(
    #             ax=ax,
    #             data=df.loc[df.k == 1],
    #             x="cost",
    #             y="energy",
    #             hue="method",
    #             size=0.0,
    #             markers=config.plot.do_markers,
    #             err_style="bars"
    #         )

    # Change spines
    # sns.despine(ax=ax, left=True, bottom=True)
    ax.spines[["right", "top", "bottom"]].set_visible(False)

    # Set X-axis scale and label
    min_x = df.cost.min()
    if config.plot.x_axis.log:
        ax.set_xscale("log")
        ax.set_xlabel(config.plot.x_axis.label + " (log)")
        ax.set_xlim([min_x, 1.0])
    else:
        ax.set_xlabel(config.plot.x_axis.label)
        ax.set_xlim([min_x, 1.0])

    # Draw line at H0
    if config.plot.do_line_top1:
        step = 0.1
        x = np.arange(ax.get_xlim()[0], ax.get_xlim()[1] + step, step)
        if config.io.data.higherbetter:
            y = df.loc[df.k == 1, "energy"].max() * np.ones(x.shape[0])
        else:
            y = df.loc[df.k == 1, "energy"].min() * np.ones(x.shape[0])
        ax.plot(x, y, linestyle=":", linewidth=2.0, color="black", zorder=1)
        ax.annotate(
            f" {opt} energy reached",
            xy=(ax.get_xlim()[0], y[0]),
            xytext=(ax.get_xlim()[0], y[0]),
            horizontalalignment="left",
            verticalalignment="bottom",
            fontsize="small",
        )

    # Set Y-label
    ax.set_ylabel(f"Mean Top-{k_plot} energy ({better} is better)")

    # Remove ticks
    ax.tick_params(axis="both", which="both", length=0)

    # Grid
    ax.grid(which="both")

    # Legend
    if not config.plot.do_all_k:
        leg_handles, leg_labels = [], []
        for handle, label in zip(leg_handles_def, leg_labels_def):
            if label in config.io.data.methods:
                leg_handles.append(handle)
                leg_labels.append(config.io.data.methods[label].name)

        leg = ax.legend(
            handles=leg_handles,
            labels=leg_labels,
            loc="best",
            title="",
            framealpha=1.0,
            frameon=True,
        )

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
    df = process_highlights(df, config)
#     df = process_diversity(df, config)
    # Plot
    fig = plot(df, config)
    # Save figure
    output_fig = output_dir / config.io.output_filename
    fig.savefig(output_fig, bbox_inches="tight")


if __name__ == "__main__":
    main()
    sys.exit()
