from gflownet.proxy.base import Proxy
import torch
from typing import List
from torchtyping import TensorType
from multiprocessing import Pool
from gflownet.utils.common import tfloat
from torch_geometric.data import Data


class AcquisitionProxy(Proxy):
    def __init__(self, acquisition, **kwargs):
        super().__init__(**kwargs)
        self.acquisition = acquisition

    def __call__(self, states):
        return self.acquisition(states)


class OCPProxy(Proxy):
    ENERGY_INVALID_SAMPLE = torch.tensor([10])

    def __init__(self, acquisition, dataset_handler, n_cpu_threads=1, **kwargs):
        super().__init__(**kwargs)
        self.acquisition = acquisition
        self.n_cpu_threads = n_cpu_threads
        self.dataset_handler = dataset_handler

        # if torch.cuda.is_available():
        #     self.proxy_device = "cuda"
        #     self.ENERGY_INVALID_SAMPLE = self.ENERGY_INVALID_SAMPLE.to(
        #         self.proxy_device
        #     )

    @torch.no_grad()
    def __call__(
        self, states: List[List[List[Data]]], verbose: bool = False
    ) -> TensorType["batch"]:
        """
        Forward pass of the proxy.

        Args:
            states (List[List[List[Data]]]): States to infer on. Shape:
                ``(batch, nr_samples, nr_adsorbates)``.
                The length of the outer list: len(states) (i.e., batch_size):
                The length of the middle list: nr of samples drawn per state (i.e., there are multiple possible ways to represent the graph; can be specified in the config, how many samples should be produced)
                The length of the third list: nr of adsorbate smiles (i.e., for each smiles a graph is built)

        Returns:
            torch.Tensor: Proxy energies. Shape: ``(batch,)``.
        """
        if self.n_cpu_threads == 1:
            # Run the evaluations in a single thread
            if verbose:
                print(f"Evaluating {len(states)} states:")
            y = [self.evaluate_state(s, i, verbose) for i, s in enumerate(states)]
        else:
            # Run the evaluations in parallel
            if verbose:
                print("Ignoring verbose flag in multi-processing")
            with Pool(processes=self.n_cpu_threads) as pool:
                y = pool.map(self.evaluate_state, states)

        return tfloat(y, self.device, self.float).view(-1)

    def evaluate_state(self, samples, index=None, verbose=False) -> float:
        """Attribute a reward score to a given state"""

        if samples is None:
            return self.ENERGY_INVALID_SAMPLE

        # Score the PyXtal samples using the proxy
        if len(samples) > 0:
            pyxtal_sample_scores = []

            # TODO : If possible, process the PyXtal samples in batches
            for sid, adsorbate_samples in enumerate(samples):
                # Convert the PyXtal crystal to the graph format expected by the model
                if verbose:
                    print(f"    Scoring PyXtal sample {sid+1}/{len(samples)}...")

                # For every sample graph invoke the GNN model
                gnn_output_per_adsorbate = []
                for sample_graph in adsorbate_samples:
                    # If the graph is invalid,
                    if sample_graph is None:
                        gnn_output_per_adsorbate.append(None)
                        continue

                    # Score the sample using the model
                    if verbose:
                        print("      gnn_output_from_graph...")
                    gnn_output = self.acq_output_from_graph(sample_graph)
                    gnn_output_per_adsorbate.append(gnn_output)

                # If the process has produced invalid graphs, mark the sample as invalid
                if None in gnn_output_per_adsorbate:
                    pyxtal_sample_scores.append(self.ENERGY_INVALID_SAMPLE)
                    continue

                # If only one adsorbate was used, this is what is used as the sample
                # score for the sample. If two adsorbates were used, we use the
                # difference (adsorbate1 - adsorbate2) as the sample score for the
                # sample.
                if len(gnn_output_per_adsorbate) == 1:
                    sample_score = gnn_output_per_adsorbate[0]
                else:
                    sample_score = (
                        gnn_output_per_adsorbate[0] - gnn_output_per_adsorbate[1]
                    )

                pyxtal_sample_scores.append(sample_score)

            global_sample_score = min(pyxtal_sample_scores)
            if verbose:
                print(f"  PyXtal global_sample_score: {global_sample_score}")
                if len(pyxtal_sample_scores) > 1:
                    print(
                        f"  Mean: {torch.mean(torch.stack(pyxtal_sample_scores)): .4f}, std: {torch.std(torch.stack(pyxtal_sample_scores)): .4f}"
                    )

        else:
            # PyXtal was unable to generate valid crystals given the state. Provide a
            # default bad score.
            if verbose:
                print("  No valid samples generated")
            global_sample_score = self.ENERGY_INVALID_SAMPLE

        if verbose:
            print()
        return global_sample_score

    def acq_output_from_graph(self, graph):
        dataset = self.dataset_handler.get_custom_dataset([graph])
        output = self.acquisition(dataset)

        return output


class OCPDiffProxy(Proxy):
    ENERGY_INVALID_SAMPLE = torch.tensor([0])

    def __init__(self, acquisition, dataset_handler, n_cpu_threads=1, **kwargs):
        super().__init__(**kwargs)
        self.acquisition = acquisition
        self.n_cpu_threads = n_cpu_threads
        self.dataset_handler = dataset_handler

        # if torch.cuda.is_available():
        #     self.proxy_device = "cuda"
        #     self.ENERGY_INVALID_SAMPLE = self.ENERGY_INVALID_SAMPLE.to(
        #         self.proxy_device
        #     )

    @torch.no_grad()
    def __call__(
        self, states: List[List[List[Data]]], verbose: bool = False
    ) -> TensorType["batch"]:
        """
        Forward pass of the proxy.

        Args:
            states (List[List[List[Data]]]): States to infer on. Shape:
                ``(batch, nr_samples, nr_adsorbates)``.
                The length of the outer list: len(states) (i.e., batch_size):
                The length of the middle list: nr of samples drawn per state (i.e., there are multiple possible ways to represent the graph; can be specified in the config, how many samples should be produced)
                The length of the third list: nr of adsorbate smiles (i.e., for each smiles a graph is built)

        Returns:
            torch.Tensor: Proxy energies. Shape: ``(batch,)``.
        """
        if self.n_cpu_threads == 1:
            # Run the evaluations in a single thread
            if verbose:
                print(f"Evaluating {len(states)} states:")
            y = [self.evaluate_state(s, i, verbose) for i, s in enumerate(states)]
        else:
            # Run the evaluations in parallel
            if verbose:
                print("Ignoring verbose flag in multi-processing")
            with Pool(processes=self.n_cpu_threads) as pool:
                y = pool.map(self.evaluate_state, states)

        return tfloat(y, self.device, self.float).view(-1)

    def evaluate_state(self, samples, index=None, verbose=False) -> float:
        """Attribute a reward score to a given state"""

        if samples is None:
            return self.ENERGY_INVALID_SAMPLE

        # Score the PyXtal samples using the proxy
        if len(samples) > 0:
            pyxtal_sample_scores = []

            # TODO : If possible, process the PyXtal samples in batches
            for sid, adsorbate_samples in enumerate(samples):
                # Convert the PyXtal crystal to the graph format expected by the model
                if verbose:
                    print(f"    Scoring PyXtal sample {sid+1}/{len(samples)}...")

                # Score the sample using the model
                if verbose:
                    print("      gnn_output_from_graph...")
                sample_score = self.acq_output_from_graph(adsorbate_samples)

                pyxtal_sample_scores.append(sample_score)

            global_sample_score = min(pyxtal_sample_scores)
            if verbose:
                print(f"  PyXtal global_sample_score: {global_sample_score}")
                if len(pyxtal_sample_scores) > 1:
                    print(
                        f"  Mean: {torch.mean(torch.stack(pyxtal_sample_scores)): .4f}, std: {torch.std(torch.stack(pyxtal_sample_scores)): .4f}"
                    )

        else:
            # PyXtal was unable to generate valid crystals given the state. Provide a
            # default bad score.
            if verbose:
                print("  No valid samples generated")
            global_sample_score = self.ENERGY_INVALID_SAMPLE

        if verbose:
            print()
        return global_sample_score

    def acq_output_from_graph(self, graph):
        dataset = self.dataset_handler.graphs2acquisition([graph])
        output = self.acquisition(dataset)

        return output
