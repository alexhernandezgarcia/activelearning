import biotite.sequence as biotite_seq
import biotite.sequence.align as align
import glob
import pandas as pd
import wandb
import itertools
import torch
import numpy as np
import os
substitution_matrix = align.SubstitutionMatrix.std_protein_matrix()

# ARGUMENTS
# k = 10
# train_dataset = "/home/mila/n/nikita.saxena/activelearning/storage/amp/sf/data_train.csv"
# test_dataset = "/home/mila/n/nikita.saxena/activelearning/storage/amp/sf/data_test.csv"
# logdir = "/network/scratch/n/nikita.saxena/logs/activelearning/2023-05-03_11-56-45"
# run_path = "nikita0209/AMP-DKL/zgoc6a5q"

k = 100
train_dataset = "/home/mila/n/nikita.saxena/activelearning/storage/amp/mf/data_train.csv"
test_dataset = "/home/mila/n/nikita.saxena/activelearning/storage/amp/mf/data_test.csv"
logdir = "/network/scratch/n/nikita.saxena/logs/activelearning/2023-05-01_23-18-47"
run_path = "nikita0209/AMP-DKL/tk7kmfj4"
is_mf = True
AL_BATCH_SIZE = 32


def find_pkl_file(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pkl'):
                return os.path.join(root, file)
    return None

# project = 

api = wandb.Api()
run = api.run(run_path)
eps = 1e-3

def get_diversity(seqs):
    seqs = [biotite_seq.ProteinSequence(seq) for seq in seqs]
    scores = []
    for pair in itertools.combinations(seqs, 2):
    # for i in range(len(seqs)):
        # for j in range(i+1, len(seqs)):
        alignment = align.align_optimal(pair[0], pair[1], substitution_matrix, local=False, max_number=1)[0]
        scores.append(align.get_sequence_identity(alignment))
    scores = torch.FloatTensor(scores)
    return torch.mean(scores)


def get_novelty(dataset_seqs, sampled_seqs):
    sampled_seqs = [biotite_seq.ProteinSequence(seq) for seq in sampled_seqs]
    dataset_seqs = [biotite_seq.ProteinSequence(seq) for seq in dataset_seqs]
    min_dists = []
    for sample in sampled_seqs:
        dists = []
        sample_repeated = itertools.repeat(sample, len(dataset_seqs))
        for s_0, x_0 in zip(sample_repeated, dataset_seqs):
             alignment = align.align_optimal(s_0, x_0, substitution_matrix, local=False, max_number=1)[0]
             dists.append(align.get_sequence_identity(alignment))
        min_dists.append(min(dists))
    min_dists = torch.FloatTensor(min_dists)
    return torch.mean(min_dists)
    # return torch.FloatTensor(min_dists)
    # for i in range(len(seqs)):
    #     for j in range(i+1, len(seqs)):
    #         alignment = align.align_optimal(seqs[i], seqs[j], substitution_matrix, local=False, max_number=1)[0]
    #         scores.append(align.get_sequence_identity(alignment))

# ITERATE OVER ALL SAMPLED CSV
# in a folder, get list of all files that have "sampled_iter" in the name

initial_train_dataset = pd.read_csv(train_dataset, index_col=0)
initial_test_dataset = pd.read_csv(test_dataset, index_col=0)

initial_train_samples = initial_train_dataset["samples"].values
initial_test_samples = initial_test_dataset["samples"].values
# or as lists add them together
initial_dataset_samples = np.concatenate([initial_train_samples, initial_test_samples]).tolist()
if is_mf==True:
    initial_dataset_samples = [sample.split(";")[0] for sample in initial_dataset_samples]

pkl_file = find_pkl_file(logdir)
culm_pkl = pd.read_pickle(pkl_file)
culm_samples = culm_pkl['cumulative_sampled_samples']
culm_energies = culm_pkl['cumulative_sampled_energies']

files_with_sampled_sequences = glob.glob(os.path.join(logdir, "data/*sampled_iter*.csv"))
oracle_maximize = True

metric_diversity = []
metric_novelty = []
metric_energy = []
metric_cost = []
mean_energy_from_wandb = run.history(keys=["mean_energy_top{}".format(k)])
mean_energy_from_wandb = mean_energy_from_wandb["mean_energy_top{}".format(k)].values

# # GET DIVERSITY
# culm_samples = []
# culm_energies = torch.tensor([])

post_al_cum_cost = run.history(keys=["post_al_cum_cost"])
# find unique values from the above series
post_al_cum_cost = np.unique(post_al_cum_cost['post_al_cum_cost'])

steps = np.arange(start = AL_BATCH_SIZE, stop = len(culm_samples), step = AL_BATCH_SIZE, dtype=int)
for idx, upper_bound in enumerate(steps):
    culm_samples_curr_iter = culm_samples[0:upper_bound]
    culm_sampled_energies_curr_iter = culm_energies[0:upper_bound]
    # culm_curr_iter = 
# for idx, file in enumerate(files_with_sampled_sequences):
#     sampled_df = pd.read_csv(file)
#     sampled_samples = sampled_df['samples'].values.tolist()
#     if is_mf==True:
#         sampled_samples = [sample.split(";")[0] for sample in sampled_samples]
    # sampled_energies = torch.tensor(sampled_df['energies'].values)
    # culm_samples += sampled_samples
    # culm_energies = torch.cat([culm_energies, sampled_energies])
    idx_topk = torch.argsort(culm_sampled_energies_curr_iter, descending=oracle_maximize)[:k].tolist()
    samples_topk = [culm_samples_curr_iter[i] for i in idx_topk]
    energies_topk = [culm_sampled_energies_curr_iter[i] for i in idx_topk]
    mean_energy_topk = torch.mean(torch.FloatTensor(energies_topk))
    diff = abs(mean_energy_topk-mean_energy_from_wandb[idx])
    if diff>eps:
        print("ERROR: energy from wandb does not match for the {}th iteration".format(idx))
    metric_energy.append(mean_energy_topk.numpy())
    mean_diversity_topk = get_diversity(samples_topk)
    mean_novelty_topk = get_novelty(sampled_seqs=samples_topk, dataset_seqs=initial_dataset_samples)
    metric_diversity.append(mean_diversity_topk.numpy())
    metric_novelty.append(mean_novelty_topk.numpy())
    metric_cost.append(post_al_cum_cost[idx])

# PLOT METRICS
reward = np.array(metric_energy)
diversity = np.array(metric_diversity)
novelty = np.array(metric_novelty)
cost = np.array(metric_cost)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(cost, reward, label="reward")
ax.scatter(cost, diversity, label="diversity")
ax.scatter(cost, novelty, label="novelty")
ax.legend()
ax.set_xlabel("cost")
ax.set_ylabel("reward")
ax.set_title("AMP")
plt.savefig(os.path.join("/home/mila/n/nikita.saxena/activelearning/playground/evaluation", "metrics.png"))
# plt.savefig(os.path.join(logdir, "metrics.png"))





# seqs should be an array of sequence represented by strings, if sequence is represented by index instead, convert it first
