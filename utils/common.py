def get_figure_plots(
    env,
    cumulative_sampled_states,
    cumulative_sampled_energies,
    picked_fidelity,
    logger,
    title,
    key,
    use_context,
):
    if (hasattr(env, "env") and hasattr(env.env, "plot_samples_frequency")) or (
        hasattr(env, "env") == False and hasattr(env, "plot_samples_frequency")
    ):
        # TODO: send samples instead and do
        # TODO: rename plot_samples to plot_states if we stick to current algo
        fig = env.plot_samples_frequency(
            cumulative_sampled_states,
            title=title,
            rescale=env.rescale,
        )
        logger.log_figure(key, fig, use_context=use_context)
    if (hasattr(env, "env") and hasattr(env.env, "plot_reward_distribution")) or (
        hasattr(env, "env") == False and hasattr(env, "plot_reward_distribution")
    ):
        fig = env.plot_reward_distribution(
            scores=cumulative_sampled_energies,
            fidelity=picked_fidelity,
            title=title,
        )
        logger.log_figure(key, fig, use_context=use_context)
