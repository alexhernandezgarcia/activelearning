"""
This script plots the topK energy with respective to the cumulative cost.
"""
import sys
from pathlib import Path

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
    df = pd.DataFrame(
        columns=["method", "seed", "energy", "cost", "diversity", "round"]
    )
    for method in config.io.data.methods:
        for seed in config.io.data.methods[method].seeds:
            logdir = (
                Path(config.root_logdir) / config.io.data.methods[method].seeds[seed].logdir
            )
            runpath = config.io.data.methods[method].seeds[seed].run_path
            energy, cost, diversity = get_performance(
                logdir,
                runpath,
                config.io.data.k,
                config.io.data.higherbetter,
                config.io.data.batch_size_al,
                config.io.data.get_diversity,
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
                }
            )
            df = pd.concat([df, df_aux], axis=0, ignore_index=True)
    if "output_csv" in config.io:
        df.to_csv(config.io.output_csv, index_label="index")
    return df


def get_performance(logdir, run_path, k, higherbetter, batch_size, get_diversity=False):
    # Read data from experiment
    f_pkl = get_pkl(logdir)
    data_dict = pd.read_pickle(f_pkl)
    cumul_samples = data_dict["cumulative_sampled_samples"]
    cumul_energies = data_dict["cumulative_sampled_energies"]
    # Read data from wandb run
    api = wandb.Api()
    run = api.run(run_path)
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
        if get_diversity:
            cumul_samples_curr_round = np.array(cumul_samples[:upper_bound])
            samples_topk = cumul_samples_curr_round[idx_topk]
            mean_diversity_topk = get_diversity(samples_topk)
        # Append to lists
        energy.append(mean_energy_topk)
        cost.append(post_al_cum_cost[idx])
        if get_diversity:
            diversity.append(mean_diversity_topk.numpy())
    if not get_diversity:
        diversity = [None for _ in range(len(energy))]
    return energy, cost, diversity


def get_diversity(seqs):
    sample_states1 = torch.tensor(seqs)
    sample_states2 = sample_states1.clone()
    dist_matrix = torch.cdist(sample_states1, sample_states2, p=2)
    dist_upper_triangle = torch.triu(dist_matrix, diagonal=1)
    dist_vector = dist_upper_triangle[dist_upper_triangle != 0]
    return dist_vector


def process_cost(df, config):
    if config.plot.x_axis.type == "fraction_budget":
        df.cost = df.cost / df.cost.max()
    return df


def min_max_errorbar(a):
    return (np.min(a), np.max(a))


def get_baselines(config):
    baselines = []
    for model in config.io.data.methods:
        if (
            "family" in config.io.data.methods[model]
            and config.io.data.methods[model].family == "baseline"
        ):
            baselines.append(model)
    return baselines


def get_order_models_families(config):
    baselines = get_baselines(config)
    order = []
    families = []
    for model in config.io.data.methods:
        order.append(config.io.data.methods[model].name)
        families.append(config.io.data.methods[model].family)
    return order, families


def get_methods(config, model, tr_factor):
    if np.isnan(tr_factor):
        method = model
    else:
        method = f"{model}-{int(tr_factor)}"
    return config.io.data.methods[method].name


def df_preprocess(df, config):
    # Select task
    df = df.loc[df.task == config.io.data.task].copy()
    # Duplicate metrics to keep original
    for metric in config.io.metrics:
        df[f"{metric}_orig"] = df[metric]
    # Make model name
    methods = []
    for model, tr_factor in zip(
        df.model_type.values, df.train_upsampling_factor.values
    ):
        methods.append(get_methods(config, model, tr_factor))
    df["methods"] = methods
    # Baselines
    baselines = get_baselines(config)
    # Test upsampling factor
    df = get_upsampling_factor_method(df)
    df = df.loc[
        df.test_upsampling_factor_method.isin(
            config.io.data.test_upsampling_factor_method.keys()
        )
    ].copy()
    return df


def get_upsampling_factor_method(df):
    upsampling_method = df.method.values
    upsampling_factor = df.test_upsampling_factor.values
    upsampling_factor_method = [
        str(factor) + "_" + method
        for factor, method in zip(upsampling_factor, upsampling_method)
    ]
    df["test_upsampling_factor_method"] = upsampling_factor_method
    return df


def get_improvement_metrics(df, config):
    for constraint in df.apply_constraint.unique():
        for factor in df.test_upsampling_factor.unique():
            for metric in config.io.metrics:
                baseline = df.loc[
                    (df.apply_constraint == 0)
                    & (df.test_upsampling_factor == factor)
                    & (df.model_type == config.io.data.baseline),
                    f"{metric}_orig",
                ].values
                assert len(baseline == 1)
                baseline = baseline[0]
                results = df.loc[
                    (df.apply_constraint == constraint)
                    & (df.test_upsampling_factor == factor)
                ][f"{metric}_orig"].values
                if config.io.metrics[metric]["higherbetter"] is None:
                    continue
                if config.io.metrics[metric]["higherbetter"]:
                    if np.isclose(baseline, 0.0):
                        raise ValueError
                    df.loc[
                        (df.apply_constraint == constraint)
                        & (df.test_upsampling_factor == factor),
                        metric,
                    ] = (
                        100 * (results - baseline) / baseline
                    )
                else:
                    if any(np.isclose(results, 0.0)):
                        raise ValueError
                    df.loc[
                        (df.apply_constraint == constraint)
                        & (df.test_upsampling_factor == factor),
                        metric,
                    ] = (
                        baseline / results
                    )
    return df


def get_palette_methods_family(families, config):
    palette_base = sns.color_palette(
        config.plot.colors.methods_family.palette,
        as_cmap=False,
        n_colors=5,
    )
    palette_base = palette_base[:2] + [palette_base[3]] + [palette_base[2]]
    if len(np.unique(families)) == 3:
        palette_base = palette_base[:2] + [palette_base[3]]
    palette_dict = {}
    palette = []
    idx = 0
    for fam in families:
        if fam not in palette_dict:
            palette_dict.update({fam: palette_base[idx]})
            idx += 1
        palette.append(palette_dict[fam])
    return palette[::-1]


def customscale_forward(x):
    return x ** (1 / 2)


def customscale_inverse(x):
    return x**2


def plot(df, config):
    plot_setup()

    fig, ax = plt.subplots(figsize=(config.plot.width, config.plot.height), dpi=config.plot.dpi)

    # Plot
    sns.lineplot(ax=ax, data=df, x="cost", y="energy", hue="method")

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
    leg_labels = [config.io.data.methods[method].name for method in config.io.data.methods]
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
    if "input_csv" in config.io and Path(config.io.input_csv).exists():
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
