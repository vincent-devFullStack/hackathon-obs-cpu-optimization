#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import resource
import struct
import sys
import time
from pathlib import Path

from cosine_naive import cosine_all_pairs_checksum_naive
from cosine_numpy import cosine_all_pairs_checksum_numpy, numpy_available


def load_metadata(path: Path) -> dict:
    meta = json.loads(path.read_text(encoding="utf-8"))
    for key in ("format", "dtype", "N", "M", "D", "E_file", "A_file"):
        if key not in meta:
            raise ValueError(f"metadata missing key: {key}")
    if meta["format"] != "cosine-benchmark-v1":
        raise ValueError(f"unsupported metadata format: {meta['format']}")
    if meta["dtype"] != "float64-le":
        raise ValueError(f"unsupported dtype: {meta['dtype']}")
    return meta


def sha256_concat(paths):
    h = hashlib.sha256()
    for path in paths:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
    return h.hexdigest()


def read_f64_file(path: Path, expected_values: int):
    payload = path.read_bytes()
    expected_bytes = expected_values * 8
    if len(payload) != expected_bytes:
        raise ValueError(f"bad size for {path}: got {len(payload)}, expected {expected_bytes}")
    return struct.unpack("<" + "d" * expected_values, payload)


def proc_threads() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Threads:"):
                    return int(line.split(":", 1)[1].strip())
    except Exception:
        return -1
    return -1


def emit(obj: dict) -> None:
    print(json.dumps(obj, separators=(",", ":")), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python cosine benchmark")
    parser.add_argument("--metadata", default=str(Path(__file__).resolve().parents[1] / "data" / "metadata.json"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--version", choices=["auto", "naive", "numpy"], default="auto")
    parser.add_argument("--expected-dataset-sha", default="")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.runs <= 0 or args.repeat <= 0:
        raise SystemExit("warmup must be >= 0, runs and repeat must be > 0")

    metadata_path = Path(args.metadata).resolve()
    meta = load_metadata(metadata_path)
    n = int(meta["N"])
    m = int(meta["M"])
    d = int(meta["D"])

    data_dir = metadata_path.parent
    e_path = data_dir / str(meta["E_file"])
    a_path = data_dir / str(meta["A_file"])

    dataset_sha = sha256_concat([e_path, a_path])
    if args.expected_dataset_sha and dataset_sha != args.expected_dataset_sha:
        raise SystemExit(
            f"dataset sha mismatch: got {dataset_sha}, expected {args.expected_dataset_sha}"
        )

    has_numpy = numpy_available()
    numpy_version = None
    if has_numpy:
        import numpy as np  # pylint: disable=import-outside-toplevel

        numpy_version = np.__version__

    selected = args.version
    if selected == "auto":
        selected = "numpy" if has_numpy else "naive"
    if selected == "numpy" and not has_numpy:
        raise SystemExit("version=numpy requested but numpy is not available")

    impl_id = f"python-{selected}"

    embeddings = read_f64_file(e_path, n * d)
    axes = read_f64_file(a_path, m * d)

    func = cosine_all_pairs_checksum_numpy if selected == "numpy" else cosine_all_pairs_checksum_naive

    if args.self_check:
        checksum = func(embeddings, axes, n, m, d)
        emit(
            {
                "type": "self_check",
                "impl": impl_id,
                "ok": True,
                "N": n,
                "M": m,
                "D": d,
                "dataset_sha256": dataset_sha,
                "checksum": float(checksum),
            }
        )
        return 0

    emit(
        {
            "type": "meta",
            "impl": impl_id,
            "N": n,
            "M": m,
            "D": d,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "runs": args.runs,
            "dataset_sha256": dataset_sha,
            "build_flags": "",
            "runtime": {
                "python_version": sys.version.split()[0],
                "numpy_available": bool(has_numpy),
                "numpy_version": numpy_version,
                "warmup_executed": int(args.warmup),
                "pid": os.getpid(),
            },
        }
    )

    for _ in range(args.warmup):
        _ = func(embeddings, axes, n, m, d)

    for run_id in range(args.runs):
        ru0 = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.perf_counter_ns()
        c0 = time.process_time_ns()
        threads0 = proc_threads()

        checksum_acc = 0.0
        for _ in range(args.repeat):
            checksum_acc += func(embeddings, axes, n, m, d)

        c1 = time.process_time_ns()
        t1 = time.perf_counter_ns()
        ru1 = resource.getrusage(resource.RUSAGE_SELF)
        threads1 = proc_threads()

        wall_ns = (t1 - t0) // args.repeat
        cpu_ns = (c1 - c0) // args.repeat

        emit(
            {
                "type": "run",
                "impl": impl_id,
                "run_id": int(run_id),
                "wall_ns": int(wall_ns),
                "cpu_ns": int(cpu_ns),
                "checksum": float(checksum_acc),
                "max_rss_kb": int(ru1.ru_maxrss),
                "ctx_voluntary": int(ru1.ru_nvcsw - ru0.ru_nvcsw),
                "ctx_involuntary": int(ru1.ru_nivcsw - ru0.ru_nivcsw),
                "minor_faults": int(ru1.ru_minflt - ru0.ru_minflt),
                "major_faults": int(ru1.ru_majflt - ru0.ru_majflt),
                "max_threads": int(max(threads0, threads1)),
            }
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
