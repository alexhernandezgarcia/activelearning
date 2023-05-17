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
from matplotlib.colors import Normalize
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.SimDivFilters import rdSimDivPickers

from utils import (
    get_dash,
    get_diversity,
    get_hue_palette,
    get_n_modes,
    get_performance,
    get_pkl,
    plot_setup,
)


def build_dataframe(config):
    if config.io.task == "dna":
        substitution_matrix = align.SubstitutionMatrix.std_nucleotide_matrix()
    elif config.io.task == "amp":
        substitution_matrix = align.SubstitutionMatrix.std_protein_matrix()
    else:
        substitution_matrix = None
    df = pd.DataFrame(
        columns=[
            "method",
            "seed",
            "energy",
            "cost",
            "diversity",
            "n_modes",
            "round",
            "k",
        ]
    )
    for method in config.io.data.methods:
        if method == "sfgfn":
            datadir = "sf"
        else:
            datadir = "mf"
        train_data_f = (
            Path(config.root_logdir)
            / config.io.data.methods[method].dirname.split("/")[0]
            / "dataset"
            / datadir
            / "data_train.csv"
        )
        df_tr = pd.read_csv(train_data_f)
        for seed in config.io.data.methods[method].seeds:
            for k in config.io.data.k:
                logdir = (
                    Path(config.root_logdir)
                    / config.io.data.methods[method].dirname
                    / config.io.data.methods[method].seeds[seed].logdir
                )
                if method == "mfbo":
                    data_dict = pd.read_pickle(logdir / "cumulative_stats.pkl")
                    runpath = None
                else:
                    runpath = get_wandb_runpath(logdir)
                    data_dict = None
                energy, cost, diversity, n_modes = get_performance(
                    logdir,
                    runpath,
                    k,
                    config.io.data.higherbetter,
                    config.io.data.batch_size_al,
                    df_tr,
                    config.io.data.do_diversity,
                    config.io.task,
                    substitution_matrix,
                    data_dict=data_dict,
                )
                n_rounds = len(energy)
                df_aux = pd.DataFrame.from_dict(
                    {
                        "method": [method for _ in range(n_rounds)],
                        "seed": [seed for _ in range(n_rounds)],
                        "energy": energy,
                        "cost": cost,
                        "diversity": diversity,
                        "n_modes": n_modes,
                        "round": np.arange(len(energy)),
                        "k": [k for _ in range(n_rounds)],
                    }
                )
                df = pd.concat([df, df_aux], axis=0, ignore_index=True)
    if "output_csv" in config.io:
        df.to_csv(config.io.output_csv, index_label="index")
    return df


def process_methods(df, config):
    return df.loc[df.method.isin(config.io.do_methods)]


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
        if "sfgfn" in df.method.unique():
            df.cost = df.cost / df.loc[df.method == "sfgfn"].cost.max()
        else:
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


def make_maximimization(df, config):
    if not config.io.data.higherbetter:
        df.energy = -1 * df.energy
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
    for method in config.plot.methods:
        palette.update(
            {
                method: sns.color_palette(
                    config.plot.methods[method].palette, as_cmap=False, n_colors=10
                )[config.plot.methods[method].palette_idx]
            }
        )
    return palette


def make_dashes(config):
    dashes = {}
    for method in config.plot.methods:
        dashes.update({method: get_dash(config.plot.methods[method].dash)})
    return dashes


def make_linewidths(config):
    linewidths = {}
    for method in config.plot.methods:
        linewidths.update({method: config.plot.methods[method].linewidth})
    return linewidths


def plot(df, config):
    if config.io.data.higherbetter:
        opt = "Max."
        better = "higher"
    else:
        opt = "Min."
        better = "lower"

    plot_setup()

    if config.io.data.do_diversity:
        fig, ax = plt.subplots(
            figsize=(config.plot.width, 1.1 * config.plot.height), dpi=config.plot.dpi
        )
    else:
        fig, ax = plt.subplots(
            figsize=(config.plot.width, config.plot.height), dpi=config.plot.dpi
        )

    palette = make_palette(config)
    dashes = make_dashes(config)
    linewidths = make_linewidths(config)

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
            style="method",
            size="method",
            estimator=config.plot.estimator,
            markers=config.plot.do_markers,
            dashes=dashes,
            sizes=linewidths,
            palette=palette,
            zorder=9,
        )
        leg_handles_def, leg_labels_def = ax.get_legend_handles_labels()

        # Scatter plot of diversity
        if config.io.data.do_diversity:
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
                palette="cividis",
                #                 palette="gist_gray",
                zorder=10,
                edgecolors="none",
                s=100,
            )
            # Colorbar
            vmin = df_means.diversity.min()
            vmax = df_means.diversity.max()
            div_norm = Normalize(vmin=vmin, vmax=vmax)
            #             div_cmap = 'gist_gray_r'
            div_cmap = "cividis_r"
            sm = plt.cm.ScalarMappable(cmap=div_cmap, norm=div_norm)
            cbar = fig.colorbar(sm, ax=ax, location="top")
            cbar.ax.set_xlabel("(-) Diversity (+)")
            cbar.ax.tick_params(axis="both", which="both", length=0)
            cbar.ax.set_xticklabels([])

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
    if config.io.task == "molecules":
        ax.set_ylabel(f"Mean Top-{k_plot} energy [eV]")
    elif config.io.task in ("branin", "hartmann"):
        ax.set_ylabel(f"Mean Top-{k_plot} score")
    else:
        ax.set_ylabel(f"Mean Top-{k_plot} energy")

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
                leg_labels.append(config.plot.methods[label].name)

        if len(config.io.do_methods) > 4:
            n_cols = 2
        else:
            n_cols = 1
        leg = ax.legend(
            handles=leg_handles,
            labels=leg_labels,
            loc="best",
            title="",
            framealpha=1.0,
            frameon=True,
            ncols=n_cols,
            columnspacing=1.0,
        )
        if config.io.task in ("amp"):
            ax.get_legend().remove()

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
    df = process_methods(df, config)
    df = process_cost(df, config)
    df = process_highlights(df, config)
    df = make_maximimization(df, config)
    #     df = process_diversity(df, config)
    # Plot
    fig = plot(df, config)
    # Save figure
    output_fig = output_dir / config.io.output_filename
    fig.savefig(output_fig, bbox_inches="tight")


if __name__ == "__main__":
    main()
    sys.exit()
