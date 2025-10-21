# general-knowledge/src/continual/average_results.py
"""
Average results from multiple runs of continual learning experiments.
This script collects JSON files in results/continual_self_edits, then computes:
  1) the overall mean accuracy at each cell (r, i) by averaging per-run means (since each run has the same number of sequences),
  2) the pooled standard deviation across all sequences
Usage: 
    python3 general-knowledge/src/continual/average_results.py
"""
import os
import json
import math
import numpy as np

def load_matrices_and_metadata(base_dir):
    mean_matrices = []
    std_matrices = []
    n_sequences_reference = None
    metadata_reference = None
    total_sequences = 0

    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.startswith("summary_") and file.endswith(".json"):
                path = os.path.join(root, file)
                with open(path, 'r') as f:
                    data = json.load(f)
                    mean_matrix = data.get("mean_over_sequences")
                    std_matrix = data.get("std_over_sequences")
                    n_sequences = data.get("n_sequences", 0)

                    if mean_matrix is not None and std_matrix is not None:
                        mean_matrices.append(np.array(mean_matrix))
                        std_matrices.append(np.array(std_matrix))
                        if n_sequences_reference is None:
                            n_sequences_reference = n_sequences
                        else:
                            assert n_sequences == n_sequences_reference, \
                                f"Mismatch in n_sequences: {n_sequences} vs {n_sequences_reference}"
                        total_sequences += n_sequences

                        # Collect metadata keys except mean/std/n_sequences
                        metadata = {k: v for k, v in data.items() if k not in ["mean_over_sequences", "std_over_sequences", "n_sequences"]}

                        if metadata_reference is None:
                            metadata_reference = metadata
                            n_sequences_reference = n_sequences
                        else:
                            # Ensure all metadata fields match
                            for key, value in metadata_reference.items():
                                assert data.get(key) == value, f"Mismatch in metadata field '{key}'"
                            # Ensure no extra metadata keys appear
                            for key in metadata.keys():
                                assert key in metadata_reference, f"Unexpected metadata field '{key}'"

    return mean_matrices, std_matrices, total_sequences, metadata_reference

def compute_average_matrix(matrices):
    if not matrices:
        raise ValueError("No valid matrices found.")
    stacked = np.stack(matrices)
    avg_matrix = np.mean(stacked, axis=0)
    return np.round(avg_matrix, 4)

def compute_pooled_std(mean_matrices, std_matrices, n_per_run):
    """
    We have multiple runs of CL experiments, each with a mean and std matrix over multiple sequences.
    For this, we can't just average the stds, since they are sample stds.
    This function computes the pooled standard deviation across all sequences, given:
      - mean_matrices:  list of shape-(RxK) arrays of per-run means,
      - std_matrices:   list of shape-(RxK) arrays of per-run sample-stds,
      - n_per_run:      number of sequences in each run (all runs assumed equal).
    
    We reconstruct, for each cell (r, i), the sum of values and sum of squares from each run:
        S_k  = n_per_run * mu_k
        SS_k = (n_per_run - 1)*(sigma_k^2) + n_per_run*(mu_k^2)
    Then we pool across runs:
        N_tot  = num_runs * n_per_run
        mu_tot = (Σ_k S_k) / N_tot
        pooled_variance = [ (Σ_k SS_k) - N_tot*(mu_tot^2 ) ] / (N_tot - 1)
        pooled_std = sqrt(pooled_variance)
    """
    num_runs = len(mean_matrices)
    # Assume all runs share the same dimensions
    R, K = mean_matrices[0].shape
    pooled_std = np.zeros((R, K))

    N_tot = num_runs * n_per_run
    for r in range(R):
        for i in range(K):
            sum_S = 0.0
            sum_SS = 0.0
            for k in range(num_runs):
                mu_k = mean_matrices[k][r, i]
                sigma_k = std_matrices[k][r, i]
                # sum of values for run k at cell (r,i)
                S_k = n_per_run * mu_k
                # sum of squares for run k at cell (r,i)
                SS_k = (n_per_run - 1) * (sigma_k ** 2) + n_per_run * (mu_k ** 2)
                sum_S += S_k
                sum_SS += SS_k

            mu_tot = sum_S / N_tot
            # pooled sample variance
            var_pooled = (sum_SS - N_tot * (mu_tot ** 2)) / (N_tot - 1)
            pooled_std[r, i] = math.sqrt(var_pooled)

    return np.round(pooled_std, 4)

def save_aggregated_results(mean_matrix, std_matrix, total_sequences, metadata, output_path):
    result = {
        "mean_over_sequences": mean_matrix.tolist(),
        "std_over_sequences": std_matrix.tolist(),
        "n_sequences": total_sequences,
    }
    result.update(metadata)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)

def main():
    base_dir = "general-knowledge/results/continual_self_edits"
    output_path = os.path.join(base_dir, "overall_results.json")
    mean_matrices, std_matrices, total_sequences, metadata = load_matrices_and_metadata(base_dir)
    avg_mean_matrix = compute_average_matrix(mean_matrices)
    avg_std_matrix = compute_average_matrix(std_matrices)
    save_aggregated_results(avg_mean_matrix, avg_std_matrix, total_sequences, metadata, output_path)

if __name__ == "__main__":
    main()