import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import hsv_to_rgb
from omegaconf.listconfig import ListConfig

import itertools
import random
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
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.SimDivFilters import rdSimDivPickers


def plot_setup():
    sns.reset_orig()
    #     sns.set(style="whitegrid")
    plt.rcParams.update({"font.family": "serif"})
    plt.rcParams.update({"font.size": 18})
    plt.rcParams.update({"scatter.edgecolors": "k"})
    plt.rcParams.update(
        {
            "font.serif": [
                "Computer Modern Roman",
                "Times New Roman",
                "Utopia",
                "New Century Schoolbook",
                "Century Schoolbook L",
                "ITC Bookman",
                "Bookman",
                "Times",
                "Palatino",
                "Charter",
                "serif" "Bitstream Vera Serif",
                "DejaVu Serif",
            ]
        }
    )


def get_dash(code):
    linestyle_tuple = {
        "solid": (1, 0),
        "dotted": (1, 1),
        "dashed": (2, 1),
        "loosely dotted": (1, 2),
        "loosely dashed": (2, 2),
    }
    return linestyle_tuple[code]


def get_hue_palette(palette, n_hues):
    if isinstance(palette, list) or isinstance(palette, ListConfig):
        palettes = [
            sns.color_palette(p, as_cmap=False, n_colors=n_hue)
            for p, n_hue in zip(palette, n_hues)
        ]
        palettes = [p for pl in palettes for p in pl]
        return palettes
    else:
        return sns.color_palette(palette, as_cmap=False, n_colors=n_hue)


def get_pkl(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".pkl"):
                return os.path.join(root, file)
    return None

def get_performance(
    logdir,
    runpath,
    k,
    higherbetter,
    batch_size,
    train_data,
    do_diversity=False,
    task=None,
    substitution_matrix=None,
    data_dict=None,
):
    if data_dict is None:
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
    else:
        cumul_samples = data_dict["cumulative_sampled_samples"]
        cumul_samples = [el for sublist in cumul_samples for el in sublist]
        cumul_energies = data_dict["cumulative_sampled_energies"]
        post_al_cum_cost = data_dict["cumulative_cost"]
    # Compute metrics from each AL round
    rounds = np.arange(
        start=batch_size, stop=len(cumul_samples), step=batch_size, dtype=int
    )
    # Catch cases where post_al_cum_cost has fewer values than number of rounds
    rounds = rounds[: len(post_al_cum_cost)]
    energy = []
    cost = []
    diversity = []
    n_modes = []
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
            n_modes_topk = get_n_modes(
                samples_topk,
                task,
                substitution_matrix,
                novelty=True,
                dataset_seqs=train_data.samples.values,
            )
        # Append to lists
        energy.append(mean_energy_topk)
        cost.append(post_al_cum_cost[idx])
        if do_diversity and k > 1:
            diversity.append(mean_diversity_topk)
            n_modes.append(n_modes_topk)
    if not do_diversity or k == 1:
        diversity = [None for _ in range(len(energy))]
        n_modes = [None for _ in range(len(energy))]
    return energy, cost, diversity, n_modes


def get_biolseq_pairwise_similarity(seq_i, seq_j, substitution_matrix):
    alignment = align.align_optimal(
        seq_i, seq_j, substitution_matrix, local=False, max_number=1
    )[0]
    return align.get_sequence_identity(alignment)


def get_biolseq_bulk_similarity(seqs, substitution_matrix, ret_square_mtx=True):
    distances = [] if not ret_square_mtx else np.empty((len(seqs), len(seqs)))
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            dist = get_biolseq_pairwise_similarity(
                seqs[i], seqs[j], substitution_matrix
            )
            if ret_square_mtx:
                distances[i, j] = dist
                distances[j, i] = dist
            else:
                distances.append(dist)
    return distances


def filter_novel_seqs(
    seqs, dataset_seqs, dist_func, novelty_thresh, substitution_matrix=None
):
    novel_seqs = []
    for seq in seqs:
        dists = []
        for dataset_seq in dataset_seqs:
            if substitution_matrix is not None:
                dist = dist_func(seq, dataset_seq, substitution_matrix)
            else:
                dist = dist_func(seq, dataset_seq)
            dists.append(dist)
            if dist > novelty_thresh:
                break
        if max(dists) <= novelty_thresh:
            novel_seqs.append(seq)
    return novel_seqs


def get_n_modes(
    seqs,
    task=None,
    substitution_matrix=None,
    novelty=False,
    dataset_seqs=None,
    novelty_thresh=None,
    cluster_thresh=None,
    dataset_format="selfies",
):
    if novelty:
        assert dataset_seqs is not None

    if task in ("amp", "dna"):

        # Remove fidelity chars
        seqs = [seq.split(";")[0] for seq in seqs]
        if dataset_seqs is not None:
            dataset_seqs = [seq.split(";")[0] for seq in dataset_seqs]

        # process generated seqs and dataset seqs
        if task in "dna":
            if cluster_thresh is None:
                cluster_thresh = 0.35
            seqs = [biotite_seq.NucleotideSequence(seq) for seq in seqs]
            if novelty:
                dataset_seqs = [
                    biotite_seq.NucleotideSequence(seq) for seq in dataset_seqs
                ]
        else:  # amp
            if cluster_thresh is None:
                cluster_thresh = 0.65
            seqs = [biotite_seq.ProteinSequence(seq) for seq in seqs]
            if novelty:
                dataset_seqs = [
                    biotite_seq.ProteinSequence(seq) for seq in dataset_seqs
                ]
        if novelty_thresh is None:
            novelty_thresh = 1 - cluster_thresh
        # find novel seqs
        if novelty:
            seqs = filter_novel_seqs(
                seqs,
                dataset_seqs,
                get_biolseq_pairwise_similarity,
                novelty_thresh,
                substitution_matrix,
            )

        # cluster the novel seqs
        if len(seqs) > 0:
            random.shuffle(seqs)
            dist_mtx = get_biolseq_bulk_similarity(seqs, substitution_matrix)
            cluster_model = LeaderClustering(
                max_radius=cluster_thresh, sparse_dist=False, precomputed_dist=True
            )

            cluster_labels = cluster_model.fit_predict(dist_mtx)
            num_modes = len(np.unique(cluster_labels))
        else:
            num_modes = 0

    elif task == "molecules":
        if cluster_thresh is None:
            cluster_thresh = 0.65
        if novelty_thresh is None:
            novelty_thresh = 1 - cluster_thresh

        # Remove fidelity chars
        seqs = [seq.split(";")[0] for seq in seqs]
        if dataset_seqs is not None:
            dataset_seqs = [seq.split(";")[0] for seq in dataset_seqs]

        # process generated mols
        smiles = [sf.decoder(seq) for seq in seqs]
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        fps = [
            rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in mols
        ]

        if novelty:
            # process dataset mols
            if dataset_format in "smiles":
                dataset_smiles = dataset_seqs
            else:
                try:
                    dataset_smiles = [sf.decoder(seq) for seq in dataset_seqs]
                except:
                    import ipdb; ipdb.set_trace()
            dataset_mols = [Chem.MolFromSmiles(smi) for smi in dataset_smiles]
            dataset_fps = [
                rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                for mol in dataset_mols
            ]

            # find novel mols
            fps = filter_novel_seqs(
                fps, dataset_fps, DataStructs.TanimotoSimilarity, novelty_thresh
            )

        # do clustering
        lp = rdSimDivPickers.LeaderPicker()
        picks = lp.LazyBitVectorPick(fps, len(fps), cluster_thresh)
        num_modes = len(picks)
    else:
        return None

    return num_modes


def get_diversity(seqs, task=None, substitution_matrix=None):
    if task == "dna":
        # Remove fidelity chars
        seqs = [seq.split(";")[0] for seq in seqs]

        seqs = [biotite_seq.NucleotideSequence(seq) for seq in seqs]
        distances = []
        for pair in itertools.combinations(seqs, 2):
            alignment = align.align_optimal(
                pair[0], pair[1], substitution_matrix, local=False, max_number=1
            )[0]
            distances.append(align.get_sequence_identity(alignment))
    elif task == "amp":
        # Remove fidelity chars
        seqs = [seq.split(";")[0] for seq in seqs]

        seqs = [biotite_seq.ProteinSequence(seq) for seq in seqs]
        distances = []
        for pair in itertools.combinations(seqs, 2):
            alignment = align.align_optimal(
                pair[0], pair[1], substitution_matrix, local=False, max_number=1
            )[0]
            distances.append(align.get_sequence_identity(alignment))
    elif task in ("molecules", "molecules_ea", "molecules_ip"):
        # Remove fidelity chars
        seqs = [seq.split(";")[0] for seq in seqs]

        smiles = [sf.decoder(seq) for seq in seqs]
        mols = [Chem.MolFromSmiles(smi) for smi in smiles]
        fps = [
            rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 2048) for mol in mols
        ]
        distances = []
        for pair in itertools.combinations(fps, 2):
            tanimotosimilarity = DataStructs.TanimotoSimilarity(pair[0], pair[1])
            distances.append(tanimotosimilarity)
    else:
        import ipdb; ipdb.set_trace()
        sample_states1 = torch.tensor(seqs)
        sample_states2 = sample_states1.clone()
        dist_matrix = torch.cdist(sample_states1, sample_states2, p=2)
        dist_upper_triangle = torch.triu(dist_matrix, diagonal=1)
        distances = dist_upper_triangle[dist_upper_triangle != 0]
        distances = distances.numpy()
    return np.mean(distances)

