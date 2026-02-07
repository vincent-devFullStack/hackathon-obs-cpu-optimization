try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def numpy_available() -> bool:
    return np is not None


def cosine_all_pairs_checksum_numpy(embeddings, axes, n: int, m: int, d: int) -> float:
    if np is None:
        raise RuntimeError("numpy is not available")

    e = np.asarray(embeddings, dtype=np.float64).reshape(n, d)
    a = np.asarray(axes, dtype=np.float64).reshape(m, d)

    e_norm = np.linalg.norm(e, axis=1)
    a_norm = np.linalg.norm(a, axis=1)

    dot = e @ a.T
    denom = np.outer(e_norm, a_norm)
    cos = np.divide(dot, denom, out=np.zeros_like(dot), where=denom != 0.0)
    return float(cos.sum(dtype=np.float64))
