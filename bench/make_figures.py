import csv
import os
import statistics
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "results.csv")
OUTDIR = os.path.join(HERE, "..", "figures")
OUTFILE = os.path.join(OUTDIR, "median_times.png")

def load_results(path: str):
    data = {"naive": [], "optimized": []}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = row["version"].strip()
            wall_ns = float(row["wall_time_ns"])
            data[v].append(wall_ns)
    return data

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    data = load_results(RESULTS)
    naive = data["naive"]
    opt = data["optimized"]

    naive_med = statistics.median(naive) / 1e6  # ms
    opt_med = statistics.median(opt) / 1e6      # ms
    speedup = naive_med / opt_med

    labels = ["naive", "optimized"]
    values = [naive_med, opt_med]

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Temps médian (ms)")
    plt.title(f"Cosine all-pairs — médianes (speedup x{speedup:.1f})")
    plt.tight_layout()
    plt.savefig(OUTFILE, dpi=200)

    print(f"Figure écrite : {OUTFILE}")

if __name__ == "__main__":
    main()
