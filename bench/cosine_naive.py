import math

def cosine_all_pairs_naive(E, A):
    N, M, D = len(E), len(A), len(E[0])
    out = [[0.0] * M for _ in range(N)]

    for i in range(N):
        for j in range(M):
            dot = 0.0
            n1 = 0.0
            n2 = 0.0
            for k in range(D):
                x = E[i][k]
                y = A[j][k]
                dot += x * y
                n1 += x * x
                n2 += y * y
            out[i][j] = 0.0 if (n1 == 0.0 or n2 == 0.0) else dot / math.sqrt(n1 * n2)

    return out
