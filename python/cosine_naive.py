import math
from typing import Sequence


def cosine_all_pairs_checksum_naive(
    embeddings: Sequence[float],
    axes: Sequence[float],
    n: int,
    m: int,
    d: int,
) -> float:
    axis_norms = [0.0] * m
    for j in range(m):
        base = j * d
        s = 0.0
        for k in range(d):
            x = axes[base + k]
            s += x * x
        axis_norms[j] = math.sqrt(s)

    checksum = 0.0
    for i in range(n):
        e_base = i * d
        emb_norm_sq = 0.0
        for k in range(d):
            x = embeddings[e_base + k]
            emb_norm_sq += x * x
        emb_norm = math.sqrt(emb_norm_sq)

        if emb_norm == 0.0:
            continue

        for j in range(m):
            a_base = j * d
            dot = 0.0
            for k in range(d):
                dot += embeddings[e_base + k] * axes[a_base + k]

            denom = emb_norm * axis_norms[j]
            if denom != 0.0:
                checksum += dot / denom

    return checksum
