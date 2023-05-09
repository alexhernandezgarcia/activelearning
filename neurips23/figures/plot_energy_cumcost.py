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
import yaml
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from utils import plot_setup, get_hue_palette, get_pkl


def get_performance(logdir, run_path, higherbetter, is_mf=False, eps=1e-3):
    pkl_file = get_pkl(logdir)
    cumul_pkl = pd.read_pickle(pkl_file)
    cumul_samples = cumul_pkl['cumulative_sampled_samples']
    cumul_energies = cumul_pkl['cumulative_sampled_energies']

    metric_diversity = []
    metric_energy = []
    metric_cost = []
    # mean_energy_from_wandb = run.history(keys=["mean_energy_top{}".format(k)])
    # mean_energy_from_wandb = mean_energy_from_wandb["mean_energy_top{}".format(k)].values
    run = api.run(run_path)
    post_al_cum_cost = run.history(keys=["post_al_cum_cost"])
    post_al_cum_cost = np.unique(post_al_cum_cost['post_al_cum_cost'])

    steps = np.arange(start = AL_BATCH_SIZE, stop = len(cumul_samples), step = AL_BATCH_SIZE, dtype=int)
    for idx, upper_bound in enumerate(steps):
        cumul_samples_curr_iter = cumul_samples[0:upper_bound]
        cumul_sampled_energies_curr_iter = cumul_energies[0:upper_bound]

        idx_topk = torch.argsort(cumul_sampled_energies_curr_iter, descending=oracle_maximize)[:k].tolist()
        samples_topk = [cumul_samples_curr_iter[i] for i in idx_topk]
        energies_topk = [cumul_sampled_energies_curr_iter[i] for i in idx_topk]
        mean_energy_topk = torch.mean(torch.FloatTensor(energies_topk))
        # diff = abs(mean_energy_topk-mean_energy_from_wandb[idx])
        # if diff>eps:
            # print("ERROR: energy from wandb does not match for the {}th iteration".format(idx))
        metric_energy.append(mean_energy_topk.numpy())
        mean_diversity_topk = get_diversity(samples_topk)
        metric_diversity.append(mean_diversity_topk.numpy())
        metric_cost.append(post_al_cum_cost[idx])

    # PLOT METRICS
    reward = np.array(metric_energy)
    diversity = np.array(metric_diversity)
    cost = np.array(metric_cost)

    return reward, diversity, cost


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
    return x ** 2


def plot(df, config):
    plot_setup()

    # Get data
    metrics = [k for k in config.io.metrics.keys()]
    methods, families = get_order_models_families(config)
    _, n_hues = np.unique(
        [int(s.split("_")[0]) for s in df.test_upsampling_factor_method.unique()],
        return_counts=True,
    )
    n_hue = np.sum(n_hues)

    # Set up figure
    fig, axes = plt.subplots(
        nrows=config.plot.n_rows,
        ncols=len(metrics) // config.plot.n_rows,
        sharey=True,
        dpi=config.plot.dpi,
        figsize=(8 * len(metrics) // config.plot.n_rows, config.plot.height),
    )
    axes = axes.flatten()

    # Auxiliary figure to get legend handles
    _, axesaux = plt.subplots(
        nrows=config.plot.n_rows,
        ncols=len(metrics) // config.plot.n_rows,
        sharey=True,
    )
    axesaux = axesaux.flatten()
    axaux = sns.pointplot(
        ax=axesaux[0],
        data=df,
        estimator=np.mean,
        order=methods,
        errorbar=None,
        x=metrics[0],
        y="methods",
        hue=config.io.data.hue.csv,
        hue_order=config.io.data.test_upsampling_factor_method.keys(),
        markers=config.plot.markers,
        color="black",
        dodge=config.plot.dodge,
        errwidth=config.plot.errwidth,
        scale=config.plot.scale,
        join=False,
    )
    leg_handles_constraints, _ = axaux.get_legend_handles_labels()

    # True plots
    for idx_metric, metric in enumerate(metrics):
        for idx_constraint, constraint in enumerate(df.apply_constraint.unique()):
            ax = sns.pointplot(
                ax=axes[idx_metric],
                data=df.loc[df.apply_constraint == constraint],
                estimator=np.mean,
                order=methods,
                errorbar=min_max_errorbar,
                x=metric,
                y="methods",
                hue=config.io.data.hue.csv,
                hue_order=config.io.data.test_upsampling_factor_method.keys(),
                palette=get_hue_palette(config.plot.colors.hue.palette, n_hues),
                markers=config.plot.markers[idx_constraint],
                dodge=config.plot.dodge,
                errwidth=config.plot.errwidth,
                scale=config.plot.scale,
                join=False,
            )
            ax.get_legend().remove()

    # Legend constraints
    leg_labels_constraints = ["Unconstrained", "Hard constraints"]
    leg1 = fig.legend(
        handles=leg_handles_constraints,
        labels=leg_labels_constraints,
        loc="upper left",
        bbox_to_anchor=(0.1, 0.95, 1.0, 0.0),
        title="",
        framealpha=1.0,
        frameon=True,
        handletextpad=-0.4,
        ncol=len(leg_labels_constraints),
    )
    # Legend
    if n_hue > 1:
        leg_handles, leg_labels = axes[0].get_legend_handles_labels()
        leg_handles = leg_handles[:n_hue]
        leg_labels = leg_labels[:n_hue]
        if config.io.data.hue.csv == "test_upsampling_factor":
            leg_labels = [el + "x" for el in leg_labels]
        if config.io.data.hue.csv == "test_upsampling_factor_method":
            leg_labels = [
                config.io.data.test_upsampling_factor_method[el] for el in leg_labels
            ]
        leg2 = fig.legend(
            handles=leg_handles,
            labels=leg_labels,
            loc="upper right",
            bbox_to_anchor=(-0.1, 0.95, 1.0, 0.0),
            title=config.io.data.hue.name,
            framealpha=1.0,
            frameon=True,
            handletextpad=-0.4,
            ncol=n_hue,
        )
    ax.add_artist(leg2)

    # Axes details
    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        # Set Y-label
        if idx == 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("")
        # Set X-label
        if config.io.metrics[metric].higherbetter is not None:
            ax.set_xlabel(
                f"Improvement of {config.io.metrics[metric].name} with respect to {methods[0]}"
            )
        else:
            ax.set_xlabel(f"{config.io.metrics[metric].name}")
        # Change spines
        sns.despine(ax=ax, left=True, bottom=True)
        # Scale of X axis
        if config.io.metrics[metric].xscale == "custom":
            ax.set_xscale(
                "function", functions=(customscale_forward, customscale_inverse)
            )
        else:
            ax.set_xscale(config.io.metrics[metric].xscale)
        # X-ticks
        xticks = ax.get_xticks()
        xticklabels = xticks
        if config.io.metrics[metric].higherbetter is not None:
            if config.io.metrics[metric].higherbetter:
                xticklabels = ["{:.1f}".format(x) + " %" for x in xticks]
            else:
                xticklabels = ["{:.1f}".format(x) + "x" for x in xticks]
        else:
            xticklabels = ["{:.1f}".format(x) for x in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, fontsize="small")
        # Y-lim
        display2data = ax.transData.inverted()
        ax2display = ax.transAxes
        _, y_bottom = display2data.transform(ax.transAxes.transform((0.0, 0.0)))
        _, y_top = display2data.transform(ax.transAxes.transform((0.0, 1.0)))
        ax.set_ylim(bottom=y_bottom, top=y_top)

        # Draw line at H1
        y = np.arange(ax.get_ylim()[1], ax.get_ylim()[0], 0.1)
        if (
            config.io.metrics[metric].higherbetter
            or config.io.metrics[metric].higherbetter is None
        ):
            x = np.zeros(y.shape[0])
        else:
            x = np.ones(y.shape[0])
        ax.plot(x, y, linestyle=":", linewidth=2.0, color="black", zorder=1)
        # Draw shaded areas
        shade_palette = get_palette_methods_family(families, config)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        width = np.abs(xlim[1] - xlim[0])
        offset = 0.05
        x0 = xlim[0] - offset * width
        width = width + 2 * offset * width
        ax.set_xlim(left=x0, right=x0 + width)
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        height_alpha = 0.9
        height = height_alpha * 1.0 / len(methods)
        margin = (1.0 - height_alpha) / (len(methods) - 1)
        for idx, method in enumerate(methods[::-1]):
            lw = 0
            if metric in ["Mass_violation"]:
                if all(
                    df.loc[df.methods == method, metric].values < 21.4 * 1e-6
                ):  # 21.4 is the mean of water content
                    lw = 3.0
            if metric in [
                "neg num per mil",
            ]:
                if all(np.isclose(df.loc[df.methods == method, metric].values, 0.0)):
                    lw = 3.0
            rect = mpatches.Rectangle(
                xy=(x0, height * idx + margin * idx),
                width=width,
                height=height,
                transform=trans,
                linewidth=lw,
                edgecolor="k",
                facecolor=shade_palette[idx],
                alpha=config.plot.shade_alpha,
                zorder=0,
            )
            ax.add_patch(rect)
    # Legend methods family
    palette_methodsfam = []
    for color in shade_palette[::-1]:
        if color not in palette_methodsfam:
            palette_methodsfam.append(color)
    methods_family = []
    for methodfam in families:
        if config.io.data.families[methodfam] not in methods_family:
            methods_family.append(config.io.data.families[methodfam])
    leg_handles = []
    for methodfam, color in zip(
        methods_family,
        palette_methodsfam,
    ):
        leg_handles.append(
            mpatches.Patch(color=color, label=methodfam, alpha=config.plot.shade_alpha)
        )
    leg2 = fig.legend(
        handles=leg_handles,
        loc="upper left",
        title="",
        bbox_to_anchor=(0.1, 0.92, 1.0, 0.0),
        framealpha=1.0,
        frameon=False,
        handletextpad=0.4,
        ncol=len(methods_family),
    )
    if n_hue > 1:
        ax.add_artist(leg1)
        ax.add_artist(leg2)

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
    # Read data and build data frames
    logdir_sf1 = Path(config.root_logdir) / config.io.data.sf[1].logdir
    runpath_sf1 = Path(config.root_logdir) / config.io.data.sf[1].run_path
    rew, div, cost = get_performance(logdir_sf1, runpath_sf1, config.io.data.higherbetter)
    df_orig = pd.read_csv(to_absolute_path(config.io.input_csv), index_col=False)
    import ipdb; ipdb.set_trace()
    # Prepare data frames for plotting
    df = df_preprocess(df_orig, config)
    df = get_improvement_metrics(df, config)
    # Plot
    fig = plot(df, config)
    # Save figure
    output_fig = output_dir / config.io.output_filename
    fig.savefig(output_fig, bbox_inches="tight")


if __name__ == "__main__":
    main()
    sys.exit()
