def get_figure_plots(
    env,
    states,
    energies,
    fidelity,
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
            states,
            title=title,
            rescale=env.rescale,
        )
        logger.log_figure(key, fig, use_context=use_context)
    if (hasattr(env, "env") and hasattr(env.env, "plot_reward_distribution")) or (
        hasattr(env, "env") == False and hasattr(env, "plot_reward_distribution")
    ):
        fig = env.plot_reward_distribution(
            scores=energies,
            fidelity=fidelity,
            title=title,
        )
        logger.log_figure(key, fig, use_context=use_context)
