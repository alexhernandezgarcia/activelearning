from gflownet.proxy.base import Proxy
import torch
from pathlib import Path
import pandas as pd


class ToyOracle(Proxy):
    # TODO: resolve the kwargs error here
    def __init__(self, oracle, noise, device, float_precision, env=None):
        super().__init__(device, float_precision)
        self.oracle = oracle
        self.noise_distribution = torch.distributions.Normal(noise.mu, noise.sigma)
        self.mu = self.oracle.mu + noise.mu
        self.sigma = self.oracle.sigma + noise.sigma
        # states = env.get_all_terminating_states()
        # noise = self.noise_distribution.sample(states.shape).to(self.device)
        # self.state_noise_dict = {
        # "state": states,
        # "noise": noise,
        # }

    def __call__(self, states):
        true_values = self.oracle(states)
        noise = self.noise_distribution.sample(true_values.shape).to(self.device)
        # noise = self.state_noise_dict[states]
        noisy_values = true_values + noise
        return noisy_values


def make_dataset(env, oracles, n_fid, device, path):
    # get_states return a list right now
    states = (
        torch.Tensor(env.env.get_uniform_terminating_states(100, seed=20))
        .to(device)
        .long()
    )
    fidelities = torch.randint(0, n_fid, (len(states), 1)).to(device)
    state_fid = torch.cat([states, fidelities], dim=1)
    states_fid_oracle = env.statetorch2oracle(state_fid)
    states_oracle = states_fid_oracle[:, :-1]
    scores = torch.zeros((states.shape[0])).to(device)
    for fid in range(n_fid):
        idx_fid = torch.where(fidelities == fid)[0]
        scores[idx_fid] = oracles[fid](states_oracle[idx_fid])
    # store states in the oracle format only
    # assume that this dataset will be used to get scores from oracle only
    states_fid_oracle_list = states_fid_oracle.detach().tolist()
    readable_states = [env.state2readable(state) for state in states_fid_oracle_list]
    df = pd.DataFrame(
        {"samples": readable_states, "scores": scores.detach().cpu().tolist()}
    )
    df.to_csv(path)
