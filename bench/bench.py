import time
import csv
import random
import numpy as np

from cosine_naive import cosine_all_pairs_naive
from cosine_numpy import cosine_all_pairs_numpy

# -----------------------
# Paramètres benchmark
# -----------------------
WARMUP = 5
RUNS = 30
REPEAT = 50  # augmente la charge CPU si nécessaire

# -----------------------
# Données synthétiques
# -----------------------
def generate_data(N=500000, M=15, D=96, seed=42):
    rng = np.random.default_rng(seed)
    E = rng.random((N, D))
    A = rng.random((M, D))
    return E.tolist(), A.tolist()

E, A = generate_data()

# -----------------------
# Fonctions de mesure
# -----------------------
def measure(func, E, A):
    # warm-up
    for _ in range(WARMUP):
        func(E, A)

    wall_times = []
    cpu_times = []

    for _ in range(RUNS):
        t0 = time.perf_counter_ns()
        c0 = time.process_time_ns()

        for _ in range(REPEAT):
            func(E, A)

        c1 = time.process_time_ns()
        t1 = time.perf_counter_ns()

        wall_times.append((t1 - t0) / REPEAT)
        cpu_times.append((c1 - c0) / REPEAT)

    return wall_times, cpu_times

# -----------------------
# Benchmark
# -----------------------
print("Running naive version...")
wall_naive, cpu_naive = measure(cosine_all_pairs_naive, E, A)

print("Running numpy version...")
wall_opt, cpu_opt = measure(cosine_all_pairs_numpy, E, A)

# -----------------------
# Résumé
# -----------------------
def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]

print("\n=== Résumé ===")
print(f"Naive  wall_time median: {median(wall_naive)/1e6:.2f} ms")
print(f"Naive  cpu_time  median: {median(cpu_naive)/1e6:.2f} ms")
print(f"Optim wall_time median: {median(wall_opt)/1e6:.2f} ms")
print(f"Optim cpu_time  median: {median(cpu_opt)/1e6:.2f} ms")
print(f"Speedup (wall): x{median(wall_naive)/median(wall_opt):.2f}")

# -----------------------
# CSV
# -----------------------
with open("results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["version", "wall_time_ns", "cpu_time_ns"])
    for w, c in zip(wall_naive, cpu_naive):
        writer.writerow(["naive", w, c])
    for w, c in zip(wall_opt, cpu_opt):
        writer.writerow(["optimized", w, c])

print("\nResults written to results.csv")
