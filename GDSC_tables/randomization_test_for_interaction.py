import cupy as cp
import numpy as np
import pandas as pd

def interaction_permutation_gpu_batched(df, factorA, factorB, value, n_perm=20000, batch_size=10, seed=42):
    """
    Batched GPU permutation test for interaction sum-of-squares.
    Computes empirical p-value for the interaction between factorA and factorB.
    batch_size controls how many permutations are processed at once to save GPU memory.
    """
    df2 = df[[factorA, factorB, value]].dropna().copy()
    df2[factorA] = df2[factorA].astype('category')
    df2[factorB] = df2[factorB].astype('category')

    A_codes = df2[factorA].cat.codes.to_numpy()
    B_codes = df2[factorB].cat.codes.to_numpy()
    Y_vals = df2[value].to_numpy(dtype=np.float64)

    levels_A = df2[factorA].cat.categories.size
    levels_B = df2[factorB].cat.categories.size
    N = len(Y_vals)

    # Move to GPU
    A_gpu = cp.asarray(A_codes, dtype=cp.int32)
    B_gpu = cp.asarray(B_codes, dtype=cp.int32)
    Y_gpu = cp.asarray(Y_vals, dtype=cp.float64)

    # Observed SSAB
    sums = cp.zeros((levels_A, levels_B), dtype=cp.float64)
    sums[A_gpu, B_gpu] = Y_gpu
    mean_A = cp.sum(sums, axis=1)
    mean_B = cp.sum(sums, axis=0)
    grand_mean = cp.sum(Y_gpu)
    SSAB_obs = cp.sum((sums - mean_A[:, None] - mean_B[None, :] + grand_mean)**2)

    # Set RNG
    cp.random.seed(seed)
    count = 0

    # Process permutations in batches
    for start in range(0, n_perm, batch_size):
        end = min(start + batch_size, n_perm)
        batch_n = end - start
        if start % 1000 == 0:
            print("Completed:", start)
        # Generate batch of shuffled B
        B_perm_batch = cp.stack([cp.random.permutation(B_gpu) for _ in range(batch_n)])

        # Compute sums for batch
        sums_batch = cp.zeros((batch_n, levels_A, levels_B), dtype=cp.float64)
        A_broadcast = cp.broadcast_to(A_gpu, (batch_n, N))
        Y_broadcast = cp.broadcast_to(Y_gpu, (batch_n, N))
        sums_batch[cp.arange(batch_n)[:, None], A_broadcast, B_perm_batch] = Y_broadcast

        # Marginals
        mean_A_batch = cp.sum(sums_batch, axis=2)
        mean_B_batch = cp.sum(sums_batch, axis=1)

        # SSAB batch
        SSAB_batch = cp.sum((sums_batch - mean_A_batch[:, :, None] - mean_B_batch[:, None, :] + grand_mean)**2, axis=(1,2))

        count += cp.sum(SSAB_batch >= SSAB_obs)

    p_value = (int(count.item()) + 1) / (n_perm + 1)
    return float(SSAB_obs.item()), float(p_value)
