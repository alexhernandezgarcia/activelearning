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
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.SimDivFilters import rdSimDivPickers

from utils import get_hue_palette, get_pkl, plot_setup, get_dash


def build_dataframe(config):
    if config.io.task == "dna":
        substitution_matrix = align.SubstitutionMatrix.std_nucleotide_matrix()
    elif config.io.task == "amp":
        substitution_matrix = align.SubstitutionMatrix.std_protein_matrix()
    else:
        substitution_matrix = None
    df = pd.DataFrame(
        columns=[
            "task",
            "al_type",
            "energy",
            "diversity",
            "n_modes",
            "k",
        ]
    )
    for task in ["branin", "hartmann", "dna", "amp", "molecules_ea", "molecules_ip"]
        with open(Path(config.root_logdir) / f"{task}.yaml", 'r') as f
            config_task = yaml.safe_load(f)
        for al_type in ["sf", "mf"]:
            for k in config_task.data.k:
            train_data_f = (
                Path(config.root_logdir)
                / task
                / "dataset"
                / al_type
                / "data_train.csv"
            )
            df_tr = pd.read_csv(train_data_f)
            if len(df_tr) < k:
                continue
            if "energies" not in in df_tr:
                continue
            if config_task.data.higherbetter:
                idx_topk = np.argsort(df.energies.values)[::-1][:k]
            else:
                idx_topk = np.argsort(df.energies.values)[:k]
            samples_topk = df.samples.values[idx_topk]
            energy_topk = np.mean(df.energies.values[idx_topk])
            diversity_topk = get_diversity(samples_topk, task, substitution_matrix)
            n_modes_topk = get_n_modes(
                samples_topk,
                task,
                substitution_matrix,
            )
            df_aux = pd.DataFrame.from_dict(
                {
                    "task": task,
                    "al_type": al_type,
                    "energy": energy_topk,
                    "diversity": diversity_topk,
                    "n_modes": n_modes_topk,
                    "k": k,
                }
            )
            df = pd.concat([df, df_aux], axis=0, ignore_index=True)
        if "output_csv" in config.io:
            df.to_csv("data/train_stats.csv", index_label="index")
        return df


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
    dataset_format="smiles",
):
    if novelty:
        assert dataset_seqs is not None

    if task in ("amp", "dna"):
        if cluster_thresh is None:
            cluster_thresh = 0.7
        if novelty_thresh is None:
            novelty_thresh = 1 - cluster_thresh

        # process generated seqs and dataset seqs
        if task in "dna":
            seqs = [biotite_seq.NucleotideSequence(seq) for seq in seqs]
            if novelty:
                dataset_seqs = [
                    biotite_seq.NucleotideSequence(seq) for seq in dataset_seqs
                ]
        else:  # amp
            seqs = [biotite_seq.ProteinSequence(seq) for seq in seqs]
            if novelty:
                dataset_seqs = [
                    biotite_seq.ProteinSequence(seq) for seq in dataset_seqs
                ]

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
                dataset_smiles = [sf.decoder(seq) for seq in dataset_seqs]
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
        raise NotImplementedError

    return num_modes


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
    elif task == "molecules":
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
        sample_states1 = torch.tensor(seqs)
        sample_states2 = sample_states1.clone()
        dist_matrix = torch.cdist(sample_states1, sample_states2, p=2)
        dist_upper_triangle = torch.triu(dist_matrix, diagonal=1)
        distances = dist_upper_triangle[dist_upper_triangle != 0]
        distances = distances.numpy()
    return np.mean(distances)



@hydra.main(config_path="./config", config_name="main", version_base=None)
def main(config):
    # Determine output dir
    if config.io.output_dir.upper() == "SLURM_TMPDIR":
        output_dir = Path(os.environ["SLURM_TMPDIR"])
    else:
        output_dir = Path(to_absolute_path(config.io.output_dir))
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
    # Build data frame or read CSV
    df = build_dataframe(config)


if __name__ == "__main__":
    main()
    sys.exit()
