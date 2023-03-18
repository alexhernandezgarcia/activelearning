import torch


class RandomSampler:
    """Samples elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, env, logger, **kwargs):
        self.env = env
        self.logger = logger

    def sample_batch(self, env, n_samples, train=False):

        if hasattr(env, "get_uniform_terminating_states"):
            states = env.get_uniform_terminating_states(n_samples)
        else:
            raise NotImplementedError(
                "Not implemented for when env does not have get_uniform_terminating_states"
            )
        return states, None

    def train(self, **kwargs):
        pass

    def evaluate(
        self, samples, energies, maximize, modes=None, dataset_states=None, **kwargs
    ):
        if maximize:
            energies = torch.sort(energies, descending=True)[0]
        else:
            energies = torch.sort(energies)[0]
        if hasattr(self.env, "get_pairwise_distance"):
            pairwise_dists = self.env.get_pairwise_distance(samples)
            pairwise_dists = torch.sort(pairwise_dists, descending=True)[0]
            if modes is not None:
                min_dist_from_mode = self.env.get_pairwise_distance(samples, modes)
                # Sort in ascending order because we want minimum distance from mode
                min_dist_from_mode = torch.sort(min_dist_from_mode, descending=False)[0]
        else:
            pairwise_dists = torch.zeros_like(energies)
            min_dist_from_mode = torch.zeros_like(energies)

        dict_topk = {}
        for k in self.logger.oracle.k:
            print(f"\n Top-{k} Performance")
            mean_energy_topk = torch.mean(energies[:k])
            mean_pairwise_dist_topk = torch.mean(pairwise_dists[:k])
            mean_min_dist_from_mode_topk = torch.mean(min_dist_from_mode[:k])

            dict_topk.update({"mean_energy_top{}".format(k): mean_energy_topk})
            dict_topk.update(
                {"mean_pairwise_distance_top{}".format(k): mean_pairwise_dist_topk}
            )
            dict_topk.update(
                {
                    "mean_min_distance_from_mode_top{}".format(
                        k
                    ): mean_min_dist_from_mode_topk
                }
            )
            if self.logger.progress:
                print(f"\t Mean Energy: {mean_energy_topk}")
                print(f"\t Mean Pairwise Distance: {mean_pairwise_dist_topk}")
                print(f"\t Mean Min Distance from Mode: {mean_min_dist_from_mode_topk}")

            self.logger.log_metrics(dict_topk, use_context=False)
