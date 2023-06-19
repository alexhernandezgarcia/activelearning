"""
Computes evaluation metrics and plots from a pre-trained GFlowNet model.
"""
from argparse import ArgumentParser
import hydra
from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf
from pathlib import Path
import torch
from torch.distributions.categorical import Categorical
import sys
from gflownet.gflownet import GFlowNetAgent, Policy
from env.mfenv import MultiFidelityEnvWrapper
from utils.multifidelity_toy import ToyOracle


def add_args(parser):
    """
    Adds command-line arguments to parser

    Returns:
        argparse.Namespace: the parsed arguments
    """
    parser.add_argument(
        "--run_path",
        default=None,
        type=str,
        help="Path to Hydra run containing config.yaml",
    )
    parser.add_argument(
        "--n_samples",
        default=None,
        type=int,
        help="Number of sequences to sample",
    )
    parser.add_argument("--device", default="cpu", type=str)
    return parser


def set_device(device: str):
    if device.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def main(args):
    # Load config
    with initialize_config_dir(
        version_base=None, config_dir=args.run_path + "/.hydra", job_name="xxx"
    ):
        config = compose(config_name="config")
        print(OmegaConf.to_yaml(config))
    # Disable wandb
    config.logger.do.online = False
    # Logger
    logger = hydra.utils.instantiate(config.logger, config, _recursive_=False)
    oracle = hydra.utils.instantiate(
        config.oracle, device=config.device, float_precision=config.float_precision
    )
    oracles = []
    # target fidelity must be the last fidelity
    # config._noise_dict = dict( sorted(config._noise_dict.items(), key=operator.itemgetter(1),reverse=True))
    for fid in range(1, 3 + 1):
        toy = ToyOracle(
            oracle,
            config._noise_dict[str(fid)],
            config.device,
            config.float_precision,
        )
        oracles.append(toy)
    env = hydra.utils.instantiate(
        config.env,
        oracle=oracle,
        device=config.device,
        float_precision=config.float_precision,
    )
    env = MultiFidelityEnvWrapper(
        env,
        n_fid=3,
        oracle=oracles,
        is_oracle_proxy=config.multifidelity.is_oracle_proxy,
    )
    # The proxy is required in the env for scoring: might be an oracle or a model
    proxy = hydra.utils.instantiate(
        config.proxy,
        logger=logger,
        env=env,
        n_fid=3,
        device=config.device,
        float_precision=config.float_precision,
        oracle=oracles,
    )
    # The proxy is passed to env and used for computing rewards
    env.proxy = proxy
    gflownet = hydra.utils.instantiate(
        config.gflownet,
        device=config.device,
        float_precision=config.float_precision,
        env=env,
        buffer=config.env.buffer,
        logger=logger,
    )
    # Load final models
    ckpt = Path(args.run_path) / config.logger.logdir.ckpts
    forward_final = [
        f for f in ckpt.glob(f"{config.gflownet.policy.forward.checkpoint}*final*")
    ][0]
    gflownet.forward_policy.model.load_state_dict(
        torch.load(forward_final, map_location=set_device(args.device))
    )
    backward_final = [
        f for f in ckpt.glob(f"{config.gflownet.policy.backward.checkpoint}*final*")
    ][0]
    gflownet.backward_policy.model.load_state_dict(
        torch.load(backward_final, map_location=set_device(args.device))
    )
    # Test GFlowNet model
    gflownet.logger.test.n = args.n_samples
    l1, kl, jsd, figs = gflownet.test()
    # Save figures
    keys = [
        "True reward and GFlowNet samples",
        "GFlowNet KDE Policy",
        "Reward KDE",
        "Frequency of Samples",
    ]
    fignames = ["samples", "kde_gfn", "kde_reward", "freq_samples"]
    output_dir = Path(args.run_path) / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    for fig, figname in zip(figs, fignames):
        output_fig = output_dir / figname
        fig.savefig(output_fig, bbox_inches="tight")


if __name__ == "__main__":
    parser = ArgumentParser()
    _, override_args = parser.parse_known_args()
    parser = add_args(parser)
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args)
    sys.exit()
