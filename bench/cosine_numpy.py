import numpy as np

def cosine_all_pairs_numpy(E, A):
    E = np.asarray(E, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)

    En = np.linalg.norm(E, axis=1, keepdims=True)
    An = np.linalg.norm(A, axis=1, keepdims=True)
    En[En == 0] = 1.0
    An[An == 0] = 1.0

    E_unit = E / En
    A_unit = A / An

    return E_unit @ A_unit.T
